# Wave identification package and example notebooks

Repository contains the below
1. wave_identify python package
2. notebooks showing realtime and analysis versions of implementation

## wave_identify package

### Prerequisites
- numpy
- xarray
- metpy

Containing the following functions
- taper_func: function to apply taper to start or both ends of data
- uz_to_qr: transform u,z to q,r
- filt_project_func: main wave filtering function, see notebooks for usage

## Example notebooks (in folder tests)

### Prerequisites
- matplotlib
- pandas

### wave_identify_fc.ipynb

Example showing real-time identification of forecast waves in ECMWF, using the supplied input data.

### wave_identify.ipynb

Example centred identification using ERA5 data, not supplied.