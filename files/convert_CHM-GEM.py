#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#ensure dask and toolz are installed!
# https://github.com/pydata/xarray/issues/4164
import xarray as xr
import numpy as np
import pandas as pd
import pyresample
import pyproj
from pyproj import Transformer
import matplotlib.pyplot as plt
import glob
from natsort import natsorted


# In[ ]:


# For converting to a 10m height for use with WN
def logu(u, zin=40, zout=10):
    z0=0.01
    newu = u*np.log(zout/z0)/ np.log(zin/z0)
    return newu


# In[ ]:


gem = xr.open_mfdataset("out.nc",combine='by_coords')


# In[ ]:


#The GEM winds were output at 40m, convert to a 10m windspeed
gem['u10'] = xr.apply_ufunc(logu,gem['u'],dask='allowed') # the GEM-CHM file has u @ 40m reference height

#zonal and meridonal components
gem['U10'] = -gem['u10']*np.sin(gem['vw_dir']*np.pi/180.)
gem['V10'] = -gem['u10']*np.cos(gem['vw_dir']*np.pi/180.)

# air temp needs to be in K
gem['t'] += 273.15


# In[ ]:


gem


# In[ ]:


#We are converting to a Lambert Conformal for Canada. Adjust as required.
#specified here and in the function below as proj_dict is needed later to build up the wrfout.nc meta data
proj_dict = {'proj':'lcc', 'lat_1':50, 'lat_2':70, 'lat_0':40, 'lon_0':-96, 'x_0':0, 'y_0':0, 'ellps':'GRS80', 'datum':'NAD83', 'units':'m'} 

def resample(gem, variable, res,nx,ny, timestep):
    
    if not isinstance(variable, list):
        variable = [variable]
        
    # Open with xarray
    xref = gem

    # Load 2D lat and lon
    lat2d = xref.gridlat_0.values
    lon2d = xref.gridlon_0.values
    
    
    # Define the original swath required by pyresample
    orig_def = pyresample.geometry.SwathDefinition(lons=lon2d, lats=lat2d)

    ####
    # Definition of the target grid
    ###

    # Definition of the Canadian LCC projection (https://spatialreference.org/ref/esri/canada-lambert-conformal-conic/)
    proj_dict = {'proj':'lcc', 'lat_1':50, 'lat_2':70, 'lat_0':40, 'lon_0':-96, 'x_0':0, 'y_0':0, 'ellps':'GRS80', 'datum':'NAD83', 'units':'m'} 

    #Name of the grid
    area_id = 'test_kan'

    # Coordinate of the upper left corner of the grid
    lat_upl = lat2d.max()
    lon_upl = lon2d.min() 


    # Conversion from WGS84 lat/lon to coordinates in the Canadian LCC projection
    transformer = Transformer.from_crs("EPSG:4326", "+proj=lcc +lat_1=50 +lat_2=70 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs",always_xy=True)
    xupl , yupl = transformer.transform(lon_upl,lat_upl)

    # Definition of the targert grid according to pyresample
    targ_def = pyresample.create_area_def(area_id, proj_dict,shape = (nx,ny),
                                 upper_left_extent = (xupl,yupl),
                                 resolution=(res, res),
                                 description='Test LCC')

    # Get the lat and lon corresponding to targ_def grid
    # Maybe needed in the wrf_out.nc file??
    lons, lats = targ_def.get_lonlats()

    #different interp methods for the regrid

    # # Nearest neighbor interpolation
    # t_nearest = pyresample.kd_tree.resample_nearest(orig_def, tgem, \
    #         targ_def, radius_of_influence=500000, fill_value=None)

    # # Bilinear interpolation
    # t_bil = pyresample.bilinear.resample_bilinear(tgem, orig_def, targ_def)
    
    resampled_gem = {}

    for var in variable:
        # Load variable to be interpolated (temperature here)
        ingem = xref[var].values[timestep,:,:]
        # IDW interpolation
        wf = lambda r: 1/r**2
        resampled_gem[var] = pyresample.kd_tree.resample_custom(orig_def, ingem,                                    
                                                                targ_def, radius_of_influence=500000, neighbours=10,                                   
                                                                weight_funcs=wf, fill_value=None)
        resampled_gem[var] = np.flip(resampled_gem[var],axis=0)

    return targ_def,resampled_gem


# In[ ]:


datetime = gem.datetime.to_pandas()
datetime = datetime.dt.strftime("%Y-%m-%d_%H:%M:%S")
datetime = datetime.values


#The strategy here is to regrid each timestep and output to a nc file that we later use the memory efficient Dask load to combine into one nc file
# this avoids having to hold it all in ram. I'm sure there is a way to do it with dask w/o the intermediatary step of the nc output, but I'm not sure how
for i, time in enumerate(datetime):
    t = [time]
    
    # convert the offset from epoch to datetime strings in the correct output
    Time=xr.DataArray(np.array(t, dtype = np.dtype(('S', 16))), dims = ['Time']) # https://github.com/pydata/xarray/issues/3407
    res_gem =2500. 
    
    # number of grids to output
    nnx=151
    nny=151

    # these are the variables you want to resample
    targ_def, resampled = resample(gem, ['t','U10','V10','HGT_P0_L1_GST'],res_gem,nnx,nny,timestep=i)


    for k,d in resampled.items():
        resampled[k] = xr.concat([xr.DataArray(d.data,dims=['south_north','west_east'])], 'Time')

    lons, lats = targ_def.get_lonlats()

    sn_dim = np.arange(0,nny)*res_gem
    we_dim = np.arange(0,nnx)*res_gem

    ds = xr.Dataset(
        resampled,
        {"Times":Time, 'south_north':sn_dim,'west_east':we_dim}  

    )

    ds = ds.rename({'t':'T2'})
    
    qcloud=xr.DataArray(np.zeros(lons.shape), dims = ['south_north','west_east']) # https://github.com/pydata/xarray/issues/3407
    qcloud=xr.concat([qcloud], 'Time')
    ds['QCLOUD'] = qcloud

    # Add longitude/latitude
    lon2d = xr.DataArray(np.flip(lons,axis=0), dims = ['south_north','west_east'])
    ds['XLON'] =lon2d 

    lat2d = xr.DataArray(np.flip(lats,axis=0), dims = ['south_north','west_east'])
    ds['XLAT'] =lat2d 
    
    ds.attrs['MAP_PROJ'] = 1 #LCC

    ds.attrs['DX'] = res_gem
    ds.attrs['DY'] = res_gem

    clon,clat = targ_def.get_lonlat(int(targ_def.shape[0]/2),int(targ_def.shape[1]/2))

    ds.attrs['CEN_LAT'] = clat
    ds.attrs['CEN_LON'] = clon

    ds.attrs['MOAD_CEN_LAT'] = proj_dict['lat_0']
    ds.attrs['STAND_LON']    = proj_dict['lon_0']
    ds.attrs['TRUELAT1']    = proj_dict['lat_1']
    ds.attrs['TRUELAT2']    = proj_dict['lat_1']

    ds.attrs['BOTTOM-TOP_GRID_DIMENSION'] = 1 # 1 (surface) z-layer,
    ds.attrs['TITLE']='WRF proxy' #required for WindNinja
    
    #output the temp files, esnure tmp dir exists
    ds.to_netcdf(f'tmp/wrfout-{i}.nc',    
             format='NETCDF4', 
                encoding={
      'Times': {
         'zlib':True, 
         'complevel':5,
         'char_dim_name':'DateStrLen'
      }
   },
             unlimited_dims={'Time':True})
    
  


# In[ ]:





# In[ ]:


files = glob.glob('tmp/wrfout-*nc')
files=natsorted(files) #ensure the sorting is reasonable


# In[ ]:


# good for testing to try only a couple files
# files = files[:10]
# files


# In[ ]:


ds = xr.concat([xr.open_mfdataset(f,combine='by_coords') for f in files],dim='Time')


# In[ ]:


ds.to_netcdf(f'wrfout-gem.nc',    
             format='NETCDF4', 
                encoding={
      'Times': {
         'zlib':True, 
         'complevel':5,
         'char_dim_name':'DateStrLen'
      }
   },
    unlimited_dims={'Time':True})


# In[ ]:


# fig = plt.figure(figsize=(5,15))

# ax = fig.add_subplot(311)
# ax.imshow(t_nearest,interpolation='nearest')
# ax.set_title("Nearest neighbor")

# ax = fig.add_subplot(312)
# ax.imshow(t_idw,interpolation='nearest')
# plt.title("IDW of square distance \n using 10 neighbors");

# ax = fig.add_subplot(313)
# ax.imshow(t_bil,interpolation='nearest')
# plt.title("Bilinear Interpolation");


# plt.show()

