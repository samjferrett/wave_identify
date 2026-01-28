#%%
import numpy as np
import xarray as xr
import sys
from metpy.calc import divergence

#%%
g=9.8
beta=2.3e-11
radea=6.371e6
spd=86400.
ww=2*np.pi/spd

# Define some parameters spefic to the methodology
latmax=24.  #   +/- latitude range over which to process data.
kmin=2      # minimum zonal wavenumber
kmax=40     # maximum zonal wavenumber
pmin=2.0      # minimum period (in days)
pmax=30.0   # maximum period (in days)
y0=6.0      # meridional trapping scale (degrees)
waves=np.array(['Kelvin','WMRG','R1','R2']) # List of wave types to output
#waves=np.array(['Kelvin','WMRG']) # List of wave types to output

y0real= 2*np.pi*radea*y0/360.0   # convert trapping scale to metres
ce=2*y0real**2*beta
g_on_c=g/ce
c_on_g=ce/g

def taper_func(nc,ntime=45,kind='start'):
    '''
    function for apply taper to start ntime time points and end ntime time points of data
    nc : xarray DataArray
    ntime : int - number of time points to taper
    kind : str - 'start' or 'both' to indicate which ends to taper
    '''
    if type(ntime)!=int:
            ntime = int(ntime)
            print('Warning: ntime has been converted to int as required')

    if kind not in ['start','both']:
        raise ValueError("kind must be 'start', 'end' or 'both'")
    
    #taper the start
    taper = xr.DataArray(np.linspace(0,1,ntime),{'time':nc.time[:ntime]})
    qdata = nc.data
    qdata[:ntime] = (nc*taper).data

    #taper the end
    if kind == 'both':
        end_taper = xr.DataArray(np.linspace(1,0,ntime),{'time':nc.time[-ntime:]})
        qdata[-ntime:] = (nc*end_taper).data

    return nc.copy(data=qdata)

def uz_to_qr(u,z) :
    '''
    transform u,z to q, r using q=z*(g/c) + u; r=z*(g/c) - u 
    '''
    q = z*g_on_c+u
    r = z*g_on_c-u

    return q,r
    
def filt_project_func(u,gh,v,
                      y0=y0,waves=waves,pmin=pmin,pmax=pmax,kmin=kmin,kmax=kmax,
                      freq=.25,taper_days=45,taper_kind='start'):
    '''
    Filtering function. u, gh and v inputs need dims 'time','pressure','latitude','longitude'
    
    Inputs
    ---
    u:      zonal wind; xarray dataarray
    gh:     geopotential; height xarray dataarray
    v:      meridional wind; xarray dataarray
    y0:     trapping scale
    waves:  list of waves to calculate
    pmin:   minimum period (in days)
    pmax:   maximum period (in days)
    kmin:   minimum zonal wavenumber
    kmax:   maximum zonal wavenumber
    c_on_g: c/g
    freq:   Frequency of data, default is for 6hrly, change to 1 for daily

    Outputs
    ---
    wave_ncs : xarray dataset containing filtered u, v, z for each wave type
        Note: dimension mode on output as follows:
        0 = Kelvin; 1 = WMRG; 2 = R1; 3 = R2
    '''
    
    #convert u,z to q,r
    q,r = uz_to_qr(u,gh)
    
    #taper 45 days, on both ends, can just do start for forecast data
    q = taper_func(q, ntime=int(taper_days*(1/freq)), kind=taper_kind)
    r = taper_func(r, ntime=int(taper_days*(1/freq)), kind=taper_kind)
    v = taper_func(v, ntime=int(taper_days*(1/freq)), kind=taper_kind)

    #perform fft
    qf = q.copy(data = np.fft.fft2 (q,axes=(0,-1)))
    rf = r.copy(data = np.fft.fft2 (r,axes=(0,-1)))
    vf = v.copy(data = np.fft.fft2 (v,axes=(0,-1)))    

    lats = qf.latitude.values
    # find size of arrays 
    nf,nz,nlats,nk=np.shape(qf)
    
    # Find frequencies and wavenumbers corresponding to pmin,pmax and kmin,kmax in coeff matrices
    f=np.fft.fftfreq(nf,freq)
    fmin=np.where((f >= 1./pmax))[0][0]
    
    if freq==1 and pmin==2:
        fmax=int(nf/2)
    else:
        fmax=(np.where((f > 1./pmin))[0][0])-1
        
    f1p=fmin
    f2p=fmax+1
    f1n = -1*fmax
    f2n = -1*fmin+1
    k1p=kmin
    k2p=kmax+1
    k1n=nk-kmax
    k2n=nk-kmin+1

    # Define the parobolic cylinder functions
    spi2=np.sqrt(2*np.pi)
    dsq=np.array([spi2,spi2,2*spi2,6*spi2]) # Normalization for the 1st 4 paroblic CF
    d=np.zeros([dsq.size,nlats])
    y=lats[:]/y0
    ysq=y**2
    d[0,:]=np.exp(-ysq/4.0)
    d[1,:]=y*d[0,:]
    d[2,:]=(ysq-1.0)*d[0,:]
    d[3,:]=y*(ysq-3.0)*d[0,:]
    
    d = xr.DataArray(d,{'mode':range(4),'latitude':qf.latitude})
    dlat=np.abs(lats[0]-lats[1])
    qf_Kel=np.zeros([nf,nz,nk],dtype=complex)
    qf_mode=np.zeros([dsq.size,nf,nz,nk],dtype=complex)
    rf_mode=np.zeros([dsq.size,nf,nz,nk],dtype=complex)
    vf_mode=np.zeros([dsq.size,nf,nz,nk],dtype=complex)
    
    # reorder the spectral coefficents to make the latitudes the last dimension
    dims = [i for i in qf.dims if i != 'latitude']+['latitude']
    qf = qf.transpose(*dims)
    rf = rf.transpose(*dims)
    vf = vf.transpose(*dims)

    for m in np.arange(dsq.size) :
        if m == 0:
            qf_Kel[f1n:f2n,:,k1p:k2p] = (qf.isel(time=slice(f1n,f2n),longitude=slice(k1p,k2p))*d.isel(mode=m)*dlat).sum('latitude')/(dsq[m]*y0)
            qf_Kel[f1p:f2p,:,k1n:k2n] = (qf.isel(time=slice(f1p,f2p),longitude=slice(k1n,k2n))*d.isel(mode=m)*dlat).sum('latitude')/(dsq[m]*y0)
        for jf,jf_mode in zip([qf,rf,vf],[qf_mode,rf_mode,vf_mode]):
            jf_mode[m,f1n:f2n,:,k1n:k2n] = (jf.isel(time=slice(f1n,f2n),longitude=slice(k1n,k2n))*d.isel(mode=m)*dlat).sum('latitude')/(dsq[m]*y0)
            jf_mode[m,f1p:f2p,:,k1p:k2p] = (jf.isel(time=slice(f1p,f2p),longitude=slice(k1p,k2p))*d.isel(mode=m)*dlat).sum('latitude')/(dsq[m]*y0)
    qf_Kel = xr.DataArray(qf_Kel,{'time':qf.time,'pressure':qf.pressure,'longitude':qf.longitude})
    qf_mode = xr.DataArray(qf_mode,{'mode':range(4),'time':qf.time,'pressure':qf.pressure,'longitude':qf.longitude})
    rf_mode = xr.DataArray(rf_mode,{'mode':range(4),'time':qf.time,'pressure':qf.pressure,'longitude':qf.longitude})
    vf_mode = xr.DataArray(vf_mode,{'mode':range(4),'time':qf.time,'pressure':qf.pressure,'longitude':qf.longitude})

    # conversion to u,v,z
    uf_wave=[]
    zf_wave=[]
    vf_wave=[]
    
    for w in range(waves.size):
        if w==0:
            #Kelvin
            uf_wave.append(0.5*qf_Kel*d.isel(mode=[w]))
            zf_wave.append(0.5*qf_Kel*d.isel(mode=[w])*c_on_g)
        elif w==1:
            #WMRG
            uf_wave.append(0.5*qf_mode.isel(mode=1)*d.isel(mode=[1]))
            zf_wave.append(0.5*qf_mode.isel(mode=1)*d.isel(mode=[1])*c_on_g)
            vf_wave.append( (vf_mode.isel(mode=0)*d.isel(mode=0)).expand_dims(dict(mode=[w])) )
        elif w==2:
            #R1
            uf_wave.append( 0.5*(qf_mode.isel(mode=2)*d.isel(mode=[2])-rf_mode.isel(mode=0)*d.isel(mode=0)) )
            zf_wave.append( 0.5*(qf_mode.isel(mode=2)*d.isel(mode=[2])+rf_mode.isel(mode=0)*d.isel(mode=0))*c_on_g )
            vf_wave.append( (vf_mode.isel(mode=1)*d.isel(mode=1)).expand_dims(dict(mode=[w])) )
        elif w==3:
            #R2
            uf_wave.append( 0.5*(qf_mode.isel(mode=3)*d.isel(mode=[3])-rf_mode.isel(mode=1)*d.isel(mode=1)) )
            zf_wave.append( 0.5*(qf_mode.isel(mode=3)*d.isel(mode=[3])+rf_mode.isel(mode=1)*d.isel(mode=1))*c_on_g )
            vf_wave.append( vf_mode.isel(mode=2)*d.isel(mode=2).expand_dims(dict(mode=[w])) )
    
    uf_wave,zf_wave,vf_wave = [xr.concat(ncs,'mode') for ncs in [uf_wave,zf_wave,vf_wave]]
    dims = ['mode','time','pressure','longitude','latitude']
    
    #can optionally return coefficients only
    #return uf_wave.transpose(*dims), zf_wave.transpose(*dims), vf_wave.transpose(*dims), qf_Kel, qf_mode, rf_mode, vf_mode#

    qf_mode = qf_mode.copy(data =np.concatenate( [[qf_Kel.values],qf_mode.values[1:]]))

    u_wave = uf_wave.copy(data=np.real(np.fft.ifft2 (uf_wave,axes=uf_wave.get_axis_num(['time','longitude']))))
    z_wave = zf_wave.copy(data=np.real(np.fft.ifft2 (zf_wave,axes=zf_wave.get_axis_num(['time','longitude']))))
    v_wave = vf_wave.copy(data=np.real(np.fft.ifft2 (vf_wave,axes=vf_wave.get_axis_num(['time','longitude']))))

    #u_wave_rmclim = u_wave - u_wave.rolling(time=30,center=False).mean()
    #v_wave_rmclim = v_wave - v_wave.rolling(time=30,center=False).mean()
    #z_wave_rmclim = z_wave - z_wave.rolling(time=30,center=False).mean()
    u_wave.name='u'
    v_wave.name='v'
    z_wave.name='z'
    wave_ncs = xr.merge([u_wave,v_wave,z_wave])

    return wave_ncs

#%%
def eg_filter():
    '''
    Example filter function call
    '''
    input_dir = '/gws/nopw/j04/forsea/users/sferrett/era5/uvz850_6h/'
    #Load file which contains u, v, z 
    input_fname = f'{input_dir}era5_uvz850_6h_1x1_*.nc'
    nc = xr.open_mfdataset(input_fname)

    #may need to adjust dims depending on input file to have dims time,pressure,latitude,longitude
    nc = nc.rename({'valid_time':'time','pressure_level':'pressure'})
    #nc = nc.rename({'lat':'latitude','lon':'longitude'}).expand_dims({'pressure':[850]}).transpose(
    #    'time','pressure','latitude','longitude')
    
    wave_ncs = filt_project_func(u=nc.u,gh=nc.z,v=nc.v,freq=.25,taper_days=45)
    
    #optional but recommended: remove tapered data
    wave_ncs = wave_ncs.isel(time=slice(4*45,-4*45))

    #save output to individual files per wave type
    for mode,wave in enumerate(waves):
        if wave=='Kelvin':
            out_nc = wave_ncs.isel(mode=mode)[['u','z']]
            out_nc.to_netcdf('/path/to/output/Kelvin_uz.nc')

            div_nc = divergence(wave_ncs.isel(mode=mode).u,xr.zeros_like(wave_ncs.isel(mode=mode).u))
            div_nc.name='divergence'
            div_nc.to_netcdf('/path/to/output/Kelvin_div.nc')
        else:
            out_nc = wave_ncs.isel(mode=mode)
            out_nc.to_netcdf(f'/path/to/output/{wave}_uz.nc')
            
