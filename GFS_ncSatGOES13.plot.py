import numpy as np
import os, sys, glob
import matplotlib.pyplot as plt
import netCDF4
from mpl_toolkits.basemap import Basemap, addcyclic
import pygrib
from cpt_convert import loadCPT
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.filters import minimum_filter, maximum_filter
import matplotlib.colors as colors


def extrema(mat,mode,window):

    """find the indices of local extrema (min and max)
    in the input array."""

    mn = minimum_filter(mat, size=window, mode=mode)
    mx = maximum_filter(mat, size=window, mode=mode)

    # (mat == mx) true if pixel is equal to the local max
    # (mat == mn) true if pixel is equal to the local in
    # Return the indices of the maxima, minima

    return np.nonzero(mat == mn), np.nonzero(mat == mx)


fdir = '/home/adrian/Desktop/roger/sat_gfs'
os.chdir(fdir)

flist = glob.glob('fnl_201505*_00.grib2')
flist.sort()

width = 10
height = 8

for f in flist:
    # fnl_20150518_00_00.grib2    GridSat-GOES.goes13.2015.05.18.0000.v01.nc

    yyyy = f[4:8]
    mm = f[8:10]
    dd = f[10:12]
    hh = f[13:15]


    if os.path.exists(fdir+'/'+f) and os.path.exists(fdir+'/GridSat-GOES.goes13.'+yyyy+'.'+mm+'.'+dd+'.'+hh+'00.v01.nc'):

        # load grb data
        grbs=pygrib.open(fdir+'/'+f)
        grb = grbs.select(name='Pressure reduced to MSL')[0]  # MSLP (Eta model reduction)
        mslp=grb.values/100
        gfslat,gfslon = grb.latlons()
        grbs.close()

        # load nc data
        ncfile = netCDF4.Dataset(fdir+'/GridSat-GOES.goes13.'+yyyy+'.'+mm+'.'+dd+'.'+hh+'00.v01.nc')
        lat  = ncfile.variables['lat'][:]
        lon  = ncfile.variables['lon'][:]
        data = ncfile.variables['ch4'][0,:,:]-273.15
        lon2d, lat2d = np.meshgrid(lon, lat)
        ncfile.close()

        # Plot data
        plt.figure(figsize=(width,height),frameon=False,dpi=300)

        m=Basemap(projection='cyl',llcrnrlon=-100, \
          urcrnrlon=-60,llcrnrlat=10,urcrnrlat=34, \
          resolution='h')

        x, y = m(lon2d, lat2d)

        cpt = loadCPT('IR4AVHRR6.cpt')
        # Makes a linear interpolation
        cpt_convert = LinearSegmentedColormap('cpt', cpt)
#
#        def cm_nhcpalettle():
#            """
#            Range of values:
#                metric: 0 to 762 millimeters
#                english: 0 to 30 inches
#            """
#            # The amount of precipitation in inches
#
#            a = np.linspace(189.8-273.15,297.4-273.15,255)
#
#            # Normalize the bin between 0 and 1 (uneven bins are important here)
#            norm = [(float(i)-min(a))/(max(a)-min(a)) for i in a]
#
#    
#            C = np.array([[252,252,252],
#                            [252,252,252],
#                            [252,252,252],
#                            [252,252,252],
#                            [248,248,248],
#                            [248,248,248],
#                            [248,248,248],
#                            [248,248,248],
#                            [244,244,244],
#                            [244,244,244],
#                            [244,244,244],
#                            [244,244,244],
#                            [240,240,240],
#                            [240,240,240],
#                            [240,240,240],
#                            [240,240,240],
#                            [236,236,236],
#                            [236,236,236],
#                            [236,236,236],
#                            [236,236,236],
#                            [232,232,232],
#                            [232,232,232],
#                            [232,232,232],
#                            [232,232,232],
#                            [228,228,228],
#                            [228,228,228],
#                            [228,228,228],
#                            [228,228,228],
#                            [224,224,224],
#                            [224,224,224],
#                            [224,224,224],
#                            [224,224,224],
#                            [220,220,220],
#                            [220,220,220],
#                            [220,220,220],
#                            [220,220,220],
#                            [216,216,216],
#                            [216,216,216],
#                            [216,216,216],
#                            [216,216,216],
#                            [252,248,248],
#                            [252,248,248],
#                            [252,248,248],
#                            [252,248,248],
#                            [252,224,224],
#                            [252,224,224],
#                            [252,224,224],
#                            [252,224,224],
#                            [252,200,200],
#                            [252,200,200],
#                            [252,200,200],
#                            [252,200,200],
#                            [252,176,176],
#                            [252,176,176],
#                            [252,176,176],
#                            [252,176,176],
#                            [252,148,148],
#                            [252,148,148],
#                            [252,148,148],
#                            [252,148,148],
#                            [252,124,124],
#                            [252,124,124],
#                            [252,124,124],
#                            [252,124,124],
#                            [252,100,100],
#                            [252,100,100],
#                            [252,100,100],
#                            [252,100,100],
#                            [252, 72, 72],
#                            [252, 72, 72],
#                            [252, 72, 72],
#                            [252, 72, 72],
#                            [252, 48, 48],
#                            [252, 48, 48],
#                            [252, 48, 48],
#                            [252, 48, 48],
#                            [252, 24, 24],
#                            [252, 24, 24],
#                            [252, 24, 24],
#                            [252, 24, 24],
#                            [252,  0,  0],
#                            [252,  0,  0],
#                            [252,  0,  0],
#                            [252,  0,  0],
#                            [224, 24,  0],
#                            [224, 24,  0],
#                            [224, 24,  0],
#                            [224, 24,  0],
#                            [200, 48,  0],
#                            [200, 48,  0],
#                            [200, 48,  0],
#                            [200, 48,  0],
#                            [176, 72,  0],
#                            [176, 72,  0],
#                            [176, 72,  0],
#                            [176, 72,  0],
#                            [148,100,  0],
#                            [148,100,  0],
#                            [148,100,  0],
#                            [148,100,  0],
#                            [124,124,  0],
#                            [124,124,  0],
#                            [124,124,  0],
#                            [124,124,  0],
#                            [100,148,  0],
#                            [100,148,  0],
#                            [100,148,  0],
#                            [100,148,  0 ],
#                            [72,176,  0 ],
#                            [72,176,  0 ],
#                            [72,176,  0 ],
#                            [72,176,  0 ],
#                            [48,200,  0 ],
#                            [48,200,  0 ],
#                            [48,200,  0 ],
#                            [48,200,  0 ],
#                            [24,224,  0 ],
#                            [24,224,  0 ],
#                            [24,224,  0 ],
#                            [24,224,  0 ],
#                            [ 0,252,  0 ],
#                            [ 0,252,  0 ],
#                            [ 0,252,  0 ],
#                            [ 0,252,  0 ],
#                            [ 0,224, 24 ],
#                            [ 0,224, 24 ],
#                            [ 0,224, 24 ],
#                            [ 0,224, 24 ],
#                            [ 0,200, 48 ],
#                            [ 0,200, 48 ],
#                            [ 0,200, 48 ],
#                            [ 0,200, 48 ],
#                            [ 0,176, 72 ],
#                            [ 0,176, 72 ],
#                            [ 0,176, 72 ],
#                            [ 0,176, 72 ],
#                            [ 0,148,100 ],
#                            [ 0,148,100 ],
#                            [ 0,148,100 ],
#                            [ 0,148,100 ],
#                            [ 0,124,124 ],
#                            [ 0,124,124 ],
#                            [ 0,124,124 ],
#                            [ 0,124,124 ],
#                            [ 0,100,148 ],
#                            [ 0,100,148 ],
#                            [ 0,100,148 ],
#                            [ 0,100,148 ],
#                            [ 0, 72,176 ],
#                            [ 0, 72,176 ],
#                            [ 0, 72,176 ],
#                            [ 0, 72,176 ],
#                            [ 0, 48,200 ],
#                            [ 0, 48,200 ],
#                            [ 0, 48,200 ],
#                            [ 0, 48,200 ],
#                            [ 0, 24,224 ],
#                            [ 0, 24,224 ],
#                            [ 0, 24,224 ],
#                            [ 0, 24,224 ],
#                            [ 0,  0,252 ],
#                            [ 0,  0,252 ],
#                            [ 0,  0,252 ],
#                            [ 0,  0,252 ],
#                            [ 0,  0,224 ],
#                            [ 0,  0,224 ],
#                            [ 0,  0,224 ],
#                            [ 0,  0,224 ],
#                            [ 0,  0,200 ],
#                            [ 0,  0,200 ],
#                            [ 0,  0,200 ],
#                            [ 0,  0,200 ],
#                            [ 0,  0,176 ],
#                            [ 0,  0,176 ],
#                            [ 0,  0,176 ],
#                            [ 0,  0,176 ],
#                            [ 0,  0,148 ],
#                            [ 0,  0,148 ],
#                            [ 0,  0,148 ],
#                            [ 0,  0,148 ],
#                            [ 0,  0,124 ],
#                            [ 0,  0,124 ],
#                            [ 0,  0,124 ],
#                            [ 0,  0,124 ],
#                            [ 0,  0,100 ],
#                            [ 0,  0,100 ],
#                            [ 0,  0,100 ],
#                            [ 0,  0,100 ],
#                            [ 0,  0, 72 ],
#                            [ 0,  0, 72 ],
#                            [ 0,  0, 72 ],
#                            [ 0,  0, 72 ],
#                            [ 0,  0, 48 ],
#                            [ 0,  0, 48 ],
#                            [ 0,  0, 48 ],
#                            [ 0,  0, 48 ],
#                            [ 0,  0, 24 ],
#                            [ 0,  0, 24 ],
#                            [ 0,  0, 24 ],
#                            [ 0,  0, 24 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ],
#                            [ 0,  0,  0 ]])
#    
#            print 'precip',len(a),len(C)
#            # Create a tuple for every color indicating the normalized position on the colormap and the assigned color.
#            COLORS = []
#            for i, n in enumerate(norm):
#                COLORS.append((n, np.array(C[i])/255.))
#        
#            # Create the colormap
#            cmap = colors.LinearSegmentedColormap.from_list("precipitation", COLORS)
#        
#            return cmap,norm,a
#    
#        cmap1,norm1,clevs1 = cm_nhcpalettle()


        # plot ir temp
        OLR=m.pcolormesh(x,y,data,cmap=cpt_convert, vmin=-103, vmax=104)
#        OLR=m.pcolormesh(x,y,data,cmap=cmap1, vmin=189.8-273.15, vmax=297.4-273.15)


        OLR.cmap.set_under((1.0, 1.0, 1.0))
        OLR.cmap.set_over( (0.0, 0.0, 0.0))

        m.colorbar(OLR)

        # plot mslp
        clevs = np.arange(900.,1100.,2.)

        x, y = m(gfslon-360,gfslat)

        P=m.contour(x, y,mslp,clevs,colors='k',linewidths=0.9)
#        P=m.contour(x, y,mslp,clevs,colors='y',linewidths=0.9)
        plt.clabel(P,inline=1,fontsize=8,fmt='%1.0f',inline_spacing=1)

        # Ploting HiLo
        mode='wrap'
        window=10
        local_min, local_max = extrema(mslp, mode, window)

        prmsl, lons = addcyclic(mslp, gfslon-360)

        xlows = x[local_min]; xhighs = x[local_max]
        ylows = y[local_min]; yhighs = y[local_max]
        lowvals = prmsl[local_min]; highvals = prmsl[local_max]
    
        # plot lows as blue L's, with min pressure value underneath.
        xyplotted = []
        # don't plot if there is already a L or H within dmin meters.
        yoffset = 0.022*(m.ymax-m.ymin)
        dmin = yoffset
    
        for x,y,p in zip(xlows, ylows, lowvals):
            if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
                dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
                if not dist or min(dist) > dmin:
                    plt.text(x,y,'B',fontsize=9,fontweight='bold',
                            ha='center',va='center',color='r')
                    plt.text(x,y-yoffset,repr(int(p)),fontsize=5,
                            ha='center',va='top',color='r',
                            bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
                    xyplotted.append((x,y))
        # plot highs as red H's, with max pressure value underneath.
        xyplotted = []
        for x,y,p in zip(xhighs, yhighs, highvals):
            if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
                dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
                if not dist or min(dist) > dmin:
                    plt.text(x,y,'A',fontsize=9,fontweight='bold',
                            ha='center',va='center',color='b')
                    plt.text(x,y-yoffset,repr(int(p)),fontsize=5,
                            ha='center',va='top',color='b',
                            bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
                    xyplotted.append((x,y))


        m.drawstates(color='gray', linewidth=0.25)
        m.drawcoastlines(color='k', linewidth=0.9)
        m.drawcountries(color='k', linewidth=0.9)
#        m.drawstates(color='w', linewidth=0.25)
#        m.drawcoastlines(color='w', linewidth=0.9)
#        m.drawcountries(color='w', linewidth=0.9)

        m.drawmeridians(range(0, 360, 10),labels=[1,0,0,1],fontsize=8, linewidth=0)
        m.drawparallels(range(-180, 180, 10),labels=[1,0,0,1],fontsize=8, linewidth=0)

        plt.title('GOES13 - Channel 4 ( Inferred Temperature '+u"\u00B0" + "C"+' )')
#        plt.show()

        plt.savefig(yyyy+'.'+mm+'.'+dd+'.'+hh+'00.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()




