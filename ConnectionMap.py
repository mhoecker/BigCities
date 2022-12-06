#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
from cartopy.io.img_tiles import OSM
from cartopy.io.img_tiles import QuadtreeTiles
import metpy
import metpy.calc
import pandas as pd
from ipywidgets import IntProgress
from IPython.display import display
import xarray as xr
import scipy as sp

# f = IntProgress(min=0, max=max_count, description="Descriptor") # instantiate the bar
# display(f) # display the bar
# f.value += 1 # signal to increment the progress bar
plt.plot(1)
plt.close('all')
plt.rcParams['lines.markersize'] = 5
plt.rcParams["figure.figsize"] = (32,16)
plt.rcParams["font.size"] = 14
plt.rcParams['savefig.bbox']="tight"
plt.rcParams['savefig.facecolor']="white"
plt.plot(1)
plt.close('all')

tiler = Stamen('watercolor')
tiler = Stamen('terrain-background')
#tiler = Stamen('terrain')
#tiler = Stamen('toner')
#tiler = OSM()
#tiler = QuadtreeTiles()


# In[2]:


## Function to calculate travel time with a start delay
def traveltime(distance,speed=100.0,wait=0.0):
    return wait*1.0+distance*1.0/speed

# Function to caculate time saved between rail as compared to car or air
# Assume a car leaves instantly at 30 kph
# Assume a train departs after 30 min at 290 kph
# Assume aircraft departs after 2 hours at 500 kph
def timesaved(distance,rail_speed=200,rail_wait=1.0/2.0,air_speed=500,air_wait=2.0,car_speed=20):
    A = traveltime(distance,speed=air_speed,wait=air_wait)
    C = traveltime(distance,speed=car_speed,wait=0)
    T = traveltime(distance,speed=rail_speed,wait=rail_wait)
    CT = (C-T)
    AT = (A-T)
    dt = np.minimum(AT,CT)
    dt = np.maximum(dt,0*dt)
    return dt

def Vcos(A,B,C):    
    #|C|^2 = |A|^2 + |B|^2 -2*|A|*|B|*cos(theta)
    #
    # ( -|C|^2 + |A|^2 + |B|^2 ) / 2*|A|*|B| = cos(theta)
    #
    return((-C*C + A*A + B*B)/(2*np.abs(A*B)))

#Convert Mi to km
# 1 mi = 1.60934 km
def km_to_mi(km):
    return km/1.60934

def mi_to_km(mi):
    return mi*1.60934

# Given lattitude and longitude calculate the great circle distance
def SimpleGreatCircle(
    lon1,
    lat1,
    lon2,
    lat2,
    R=6371):
    if (lat1!=lat2) or (lon1!=lon2):
        lat1r = np.pi*lat1/180.
        lon1r = np.pi*lon1/180.
        lat2r = np.pi*lat2/180.
        lon2r = np.pi*lon2/180.
        dr = np.arccos( np.sin(lat1r)*np.sin(lat2r)+np.cos(lat1r)*np.cos(lat2r)*np.cos(lon1r-lon2r) )*R
        dr = np.abs(dr)
    else:
        dr = 0
    return dr
    

# for each city in the list do the distance calculation 
# this could be faster (N^2 opperations versus N*(N-1)/2) 
# but that might miss a connection
def CityDistances(Cities):
    N_Cities = len(Cities)
    Progress = IntProgress(
        min=0, max=N_Cities, description="Distances")
    display(Progress) # display the bar

    dr_matrix = -np.ones([N_Cities,N_Cities])
    for i in range(N_Cities):
        dr_matrix[i,i]=0.0
        Progress.value += 1
        for j in range(i):
            dr = SimpleGreatCircle(Cities.iloc[i]["Lon"],
                                   Cities.iloc[i]["Lat"],
                                   Cities.iloc[j]["Lon"],
                                   Cities.iloc[j]["Lat"])
            dr_matrix[i,j] = dr
            dr_matrix[j,i] = dr
    Progress.close()
    return dr_matrix

def dist_to_bigger(cities,dr_matrix=None):
    cities["Larger Neighbor"] = cities["Cities"]
    cities["Prominence"] = np.inf    
    if dr_matrix is None:
        dr_matrix = CityDistances(Cities)
    for idx,cityA in cities.iterrows():
        drinv = 0.0
        jkeep = idx
        for jdx,cityB in cities.iterrows():
            if cityB["City Pop"]>cityA["City Pop"]:
                if drinv*dr_matrix[idx,jdx]<1:
                    drinv = 1./dr_matrix[idx,jdx]
                    jkeep = jdx
#        print(cities["Cities"][idx],"<=",cities["Cities"][jkeep])
        cities.loc[idx,"Larger Neighbor"] = cities.loc[jkeep,"Cities"]
        if drinv!=0:
            cities.loc[idx,"Prominence"] = 1.0/drinv
    return cities, dr_matrix
# Gravity-like model
# Weight each city pair by the geometric mean of the populations
# Weight each pair by 1/distance
# If the time saved versus car or air (whichever is faster) give zero weight
def gravitymodel(Pop1,Pop2,dr):
    g = (np.sqrt(Pop1*Pop2)/dr)
    return np.sign(timesaved(dr))*g
    
def RouteWeights(dr_matrix):
    N_Cities = len(dr_matrix[0,:])
    Progress = IntProgress(
        min=0, max=N_Cities, description="Weights")
    display(Progress) # display the bar
    w_matrix = 0*dr_matrix
    for i in range(len(w_matrix[0,:])):
        Progress.value += 1
        for j in range(i):
            weight = gravitymodel(Cities.iloc[i]["City Pop"],
                                    Cities.iloc[j]["City Pop"],
                                    dr_matrix[i,j])
            w_matrix[i,j] = weight
            w_matrix[j,i] = weight
    w_matrix[w_matrix<0] = 0
    Progress.close()
    return w_matrix, dr_matrix

def makeRoutes(Cities,w_matrix=None,dr_matrix=None):
    if dr_matrix is None:
        dr_matrix = CityDistances(Cities)
    if w_matrix is None:
        w_matrix, dr_matrix = RouteWeights(dr_matrix)
    Routedict = {"Weight":[],"Large City":[],"Small City":[],"Large Index":[],"Small Index":[],"Distance":[]}
    N_Cities = len(dr_matrix[0,:])
    N_Routes = np.sum(np.sign(w_matrix.flatten()))/2
    Progress = IntProgress(
        min=0, max=N_Routes, description="Routing")
    display(Progress) # display the bar

    for i in range(len(w_matrix[0,:])):
        js = np.argsort(-w_matrix[i,:])
        for j in js:
            if (Cities.iloc[i]["City Pop"]>Cities.iloc[j]["City Pop"]) and (w_matrix[i,j]>0) and (i!=j):
                Progress.value += 1
                Routedict["Weight"].append(w_matrix[i,j])
                Routedict["Large City"].append(Cities.iloc[i]["Cities"])
                Routedict["Small City"].append(Cities.iloc[j]["Cities"])
                Routedict["Large Index"].append(i)
                Routedict["Small Index"].append(j)
                Routedict["Distance"].append(dr_matrix[i,j])
    Progress.close()
    Routes = pd.DataFrame(data=Routedict)
    return Routes, w_matrix, dr_matrix

# Lets remove some routes with obvious stopovers
# what is an obvious stopover?
# A city in roughly the same direction
#
# The distance from i to j 
# is roughly the same as it is via k
# dr[i,j] ~ dr[i,k]+dr[k,j]
#
# The angle between A = i->k and B = k->j is > pi/2
# or cos < 0
#
# So a city with cos ~ -1 and dr[i,k]<dr[i,j] and dr[j,k]<dr[i,j]
# can be added to a route with the path i->j replaced with i->k->j


# In[3]:


def RemoveRoutes(Cities,dr_matrix=None,w_matrix=None,Routes=None,Routeids=None,verbose=False):
    if dr_matrix is None:
        dr_matrix = CityDistances(Cities)
    if w_matrix is None:
        w_matrix, dr_matrix = RouteWeights(dr_matrix)
    if Routes is None:
        Routes, w_matrix, dr_matrix = makeRoutes(Cities,w_matrix,dr_matrix)
    Progress = IntProgress(
        min=0, max=len(Routeids), description="Removing")
    display(Progress) # display the bar

    for Routeid in Routeids:
        a = Routes["Large Index"][Routeid]
        b = Routes["Small Index"][Routeid]
        c = None
        drc = np.inf
        Progress.value += 1
        if verbose:
            print("Removing",Routes[1][Routeid],"to",Routes[2][Routeid])
        for j in range(len(dr_matrix[0,:])):
            if (w_matrix[a,j]>0) and (w_matrix[j,b]>0) and (j!=a) and (j!=b):
                dr = (dr_matrix[a,j]+dr_matrix[b,j])/dr_matrix[a,b]
                dw = w_matrix[a,b]/dr
                if dr<drc:
                    c = j
                    drc = dr
        if c is not None:
            if verbose:
                print("Between\n",
                    Routes["Large City"][Routeid],
                    "and",
                    Routes["Small City"][Routeid])
                print("Proposing a stop at",
                      Cities["Cities"].iloc[c],
                      "\ndr=",
                      drc
                     )
#            weight = gravitymodel(Cities.iloc[a]["City Pop"],
#                                    Cities.iloc[b]["City Pop"],
#                                    dr_matrix[a,c]+dr_matrix[b,c],
#                                    p=1)
            w_matrix[c,b] = w_matrix[a,b]/drc+w_matrix[c,b]
            w_matrix[b,c] = w_matrix[a,b]/drc+w_matrix[b,c]
            w_matrix[a,c] = w_matrix[a,b]/drc+w_matrix[a,c]
            w_matrix[c,a] = w_matrix[a,b]/drc+w_matrix[c,a]
        w_matrix[a,b] = 0
        w_matrix[b,a] = 0
    Progress.close()
    Routes_out, w_matrix_out, dr_matrix = makeRoutes(Cities=Cities,w_matrix=w_matrix,dr_matrix=dr_matrix)
    return Routes_out, w_matrix_out, dr_matrix
def TrimRoutes(Cities,
               dr_matrix=None,
               weights=None,
               Routes=None,
               dr0=None,
               useweights=0,
               verbose=False,
               very_verbose=False):
    if dr_matrix is None:
        dr_matrix = CityDistances(Cities)
    if weights is None:
        w_matrix, dr_matrix = RouteWeights(dr_matrix)
    if Routes is None:
        Routes, w_matrix, dr_matrix = makeRoutes(Cities,weights,dr_matrix)
    else:
        w_matrix = 1.0*weights
    if dr0 is None:
        dr0 = 4
    if verbose:
        print("Number of routes before =",len(Routes[0]))
    RouteOrder = np.argsort(Routes["Distance"])[::-1]
    Progress = IntProgress(
        min=0, max=len(RouteOrder), description="Trimming")
    display(Progress) # display the bar

    for i in RouteOrder:
        a = Routes["Large Index"][i]
        b = Routes["Small Index"][i]
        c = None
        drc = dr0
        Progress.value += 1
        for j in range(len(dr_matrix[0,:])):
            if (w_matrix[a,j]>0) and (w_matrix[j,b]>0) and (j!=a) and (j!=b):
                dr = (dr_matrix[a,j]+dr_matrix[b,j])/dr_matrix[a,b]
                dw  = w_matrix[a,b]/dr
                if useweights >= 2 :
                    keep = (w_matrix[a,b]-w_matrix[a,j])<=dw and (w_matrix[a,b]-w_matrix[b,j])<=dw
                    keep = keep and dr<=drc
                elif useweights >= 1:
                    keep = ((w_matrix[a,b]-w_matrix[a,j])<=dw or (w_matrix[a,b]-w_matrix[b,j])<=dw)
                    keep = keep and dr<=drc
                else:
                    keep = dr<=drc
                if keep:
                    c = j
                    drc = dr
                    dwc = dw
        if c is not None:
            if very_verbose:
                print("Between\n",
                    Routes["Large City"][i],
                    "and",
                    Routes["Small City"][i])
                print("Proposing a stop at",
                      Cities["Cities"].iloc[c],
                      "\ndr=",
                      drc,
                      "dw =",
                      dwc
                     )
            w_matrix[c,b] = dwc+w_matrix[c,b]
            w_matrix[b,c] = dwc+w_matrix[b,c]
            w_matrix[a,c] = dwc+w_matrix[a,c]
            w_matrix[c,a] = dwc+w_matrix[c,a]
            w_matrix[a,b] = 0
            w_matrix[b,a] = 0

    Progress.close()
    Routes, w_matrix, dr_matrix = makeRoutes(Cities=Cities,
                                             w_matrix=w_matrix,
                                             dr_matrix=dr_matrix)
    if verbose:
        print("Number of routes after =",len(Routes[0]))
    return Routes , w_matrix, dr_matrix

def route_index(Routes,
                cityA="Seattle-Tacoma-Bellevue",
                cityB="Bremerton-Silverdale-Port Orchard"):
    j = None
    for i in range(len(Routes[0])):
        hasA = (cityA == Routes[1][i]) or (cityA == Routes[2][i])
        hasB = (cityB == Routes[1][i]) or (cityB == Routes[2][i])
        if hasA and hasB:
            j = i
    return j
def remove_route(Routes,
                 weights,
                 cityA="Seattle-Tacoma-Bellevue",
                 cityB="Bremerton-Silverdale-Port Orchard"):
    Ridx = route_index(Routes,cityA=cityA,cityB=cityB)
    if Ridx is not None:
        i = Routes[3][Ridx]
        j = Routes[4][Ridx]
        weights[i,j] = 0
        weights[j,i] = 0
    return weights


# In[4]:


x = np.arange(8,1000,1)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x,gravitymodel(1.0,1.0,x))
#ax.semilogx(x,gravitymodel(1.0,1.0,x,1))
ax2 = ax.twiny()
miles = np.arange(0,km_to_mi(x.max()),50)
ax2.set_xticks(mi_to_km(miles))
ax2.set_xticklabels(["%4.0f" % z for z in miles])
ax2.set_xlim(ax.get_xlim())


# In[5]:


def refiner(x,y,z,upscale=12,kind='linear'):
    if upscale!=1 and upscale>0:
        xo = np.arange(x[0], x[-1], (x[1]-x[0])/(upscale+1))
        yo = np.arange(y[0], y[-1], (y[1]-y[0])/(upscale+1))
        interp = sp.interpolate.interp2d(x,y,z,kind)
        zi = interp(xo,yo)
        zo = sp.ndimage.gaussian_filter(zi,0.25*upscale)
        return xo, yo, zo
def renorm(array):
    out = array
    out = out-np.nanmin(out)
    out = out/np.nanmax(out)
    out[np.isnan(out)] = 0
    return out
NLADS = xr.open_dataset("NLDAS_elevation.nc4")

slope = np.squeeze(NLADS.NLDAS_slope.values)
slope = renorm(slope)
X,Y,slope = refiner(NLADS.lon.values, NLADS.lat.values,slope)
slope = renorm(slope)
#plt.imshow(slope)

elev = np.squeeze(NLADS.NLDAS_elev.values)
elev = renorm(elev)
#G0 = sp.ndimage.sobel(elev,axis=0)
#G1 = sp.ndimage.sobel(elev,axis=1)
#elev = np.sqrt(G0**2+G1**2)
X,Y,elev = refiner(NLADS.lon.values, NLADS.lat.values,elev)
elev = renorm(elev)
#plt.imshow(elev)

estd = np.squeeze(NLADS.NLDAS_elev_std.values)
estd = renorm(estd)
X,Y,estd = refiner(NLADS.lon.values, NLADS.lat.values,estd)
estd = renorm(estd)
#plt.imshow(estd)


# In[6]:


#Load in the city data
# a set of colon separated coloums with city populations, latitudes and longitudes
# 
# City Name : Population : Latitude : Longitude
#
AllCities = pd.read_csv("test500k.csv",sep=",",header=0)
AllCities[["Cities","States"]] = AllCities["United States"].str.split(",",expand=True)
AllCities["States"] = AllCities['States'].str.replace(r'Metro Area', '')
AllCities["States"] = AllCities['States'].str.replace(r' ', '',regex=True)
AllCities["Cities"] = AllCities['Cities'].str.replace(r'.', '',regex=True)
Cities = AllCities
print(pd.unique(Cities["States"]))
#Cities = AllCities[AllCities["States"].isin(["MI","WI","IL","IN","IA",'OH',"MO",'IL-IN-WI','KY-IN','MO-IL','MO-KS','MN-WI','WI-MN','OH-KY-IN'])]
#Cities = AllCities[AllCities["States"].isin(["FL"])]
#Cities = AllCities[AllCities["States"].isin(["NJ","NY","DE","MD","DC","VA","WV","PA","NC","SC",'NC-SC','PA-NJ','VA-WV','MD-WV','MD-DE',
#                                             'NY-NJ-PA','DC-VA-MD-WV','PA-NJ-DE-MD',])]
#Cities = AllCities[AllCities["States"].isin(["CA"])]
#Cities = AllCities[AllCities["States"].isin(["WA","OR-WA","OR","ID","MT"])]
#Cities = AllCities[AllCities["States"].isin(["WA","OR-WA","OR","NV","CA","AZ","UT","ID"])]
#Cities = AllCities[AllCities["States"].isin(["CA","WA","OR-WA","NV","AZ","WA","OR","ID","UT","CO","NM","MT","TX","KS","OK","SD","ND",'IA-NE-SD','TX-AR','ND-MN','AR-OK','NE','NE-IA','MO-KS'])]
Cities = Cities.reset_index(drop=True)

Cities, dr_matrix = dist_to_bigger(Cities)
Cities["Pop rank"] = np.argsort(-Cities["City Pop"].values)
Cities["Pop percentile"] = (np.argsort(Cities["City Pop"].values))/(len(Cities["City Pop"].values)-1)
#for city in Cities.values:
#    print(city[1])


# In[7]:


##### Set up a globe with a specific radius
globe = ccrs.Globe(semimajor_axis=6371000.)

# Set up a Orthographic
#proj = ccrs.Orthographic(
#    central_longitude=-110.0, 
#    central_latitude=45.0, 
#    globe=None)
# Setup a projection for longitude/latitude coordinates
XYproj = ccrs.PlateCarree()
GCproj = ccrs.Geodetic()
# Set the extent for the maps
CONUS_Extent = np.array([-123.0, -69.5, 25.25, 49.25])
#West_Extent = [-122, -105, 31.75, 48.75]
#Central_Extent = [-101,-88,28.0,45.5]
#East_Extent = [-90, -74, 25.5, 44.5]
#NW_Extent =       [-125.0, -105.0, 37.0, 49.0]
#SW_Extent =       [-120.0, -100.0, 25.0, 37.0]
#MtNorth_Extent =  [-110.0,  -90.0, 37.0, 49.0]
#MtSouth_Extent =  [-105.0,  -95.0, 25.0, 37.0]
#NCentral_Extent = [ -95.0,  -75.0, 37.0, 49.0]
#SCentral_Extent = [-100.0,  -80.0, 25.0, 37.0]
#NE_Extent =       [ -80.0,  -60.0, 37.0, 49.0]
#SE_Extent =       [ -85.0,  -65.0, 25.0, 37.0]
Extents = [CONUS_Extent]
files = ["CONUS.png"]
Nx = 5
Ny = 4
dx = np.abs(CONUS_Extent[1]-CONUS_Extent[0])/Nx
dy = np.abs(CONUS_Extent[3]-CONUS_Extent[2])*1.0/Ny
print(dx,dy)
for i in range(Nx):
    for j in range(Ny):
        extent = np.array([
                CONUS_Extent[0]+i*dx,
                CONUS_Extent[0]+(1.0+i)*dx,
                CONUS_Extent[2]+j*dy,
                CONUS_Extent[2]+(j+1.0)*dy])
        if i>0:
            extent[0] = extent[0]-1.0*dx*(Nx-1.0)/Nx
        if i+1<Nx:
            extent[1] = extent[1]+1.0*dx*(Nx-1.0)/Nx
        if j>0:
            extent[2] = extent[2]-1.0*dy*(Ny-1.0)/Ny
        if j+1<Ny:
            extent[3] = extent[3]+1.0*dy*(Ny-1.0)/Ny
#        print(i,j,extent)
        Extents.append(extent)
        files.append(str(i).zfill(2)+"-"+str(j).zfill(2)+".png")

for extent,file in zip(Extents,files):
    print(file,extent)
#proj = ccrs.LambertCylindrical(central_longitude=np.mean(CONUS_Extent[0:2]))
#proj = ccrs.Mercator(central_longitude=np.mean(CONUS_Extent[0:2]),
#                  min_latitude=np.min(CONUS_Extent[2:4])-5 )
#                  max_latitude=np.min(CONUS_Extent[2:4]) )
#                  globe=None,
#                  latitude_true_scale=np.mean(CONUS_Extent[2:4]))
proj = ccrs.AlbersEqualArea(
    central_longitude=np.mean(CONUS_Extent[0:2]), 
    central_latitude=np.mean(CONUS_Extent[2:4]), 
    standard_parallels=(CONUS_Extent[2], CONUS_Extent[3]))
#    proj = ccrs.Orthographic(
#        central_longitude=np.mean(extent[0:2]), 
#        central_latitude=np.mean(extent[2:4]), 
#        globe=None)
#    print(proj)


# In[8]:


plt.close('all')
fig = plt.figure()
zmap = cm.get_cmap('Greens_r')
smap = cm.get_cmap('Greys')
N = len(Extents)
#N = 1
Nsqrt = int(np.ceil(np.sqrt(N)))
for i in range(N):
    #print(files[i],Extents[i])
    proj = proj
    ax = fig.add_subplot(Nsqrt, Nsqrt, i+1, projection=proj)
    #ax = fig.add_subplot(1, 1, 1, projection=proj)
    #ax = fig.add_subplot(Nsqrt, Nsqrt, i+1)
    # Sets the extent using a lon/lat box
    ax.margins(tight=True)
    ax.set_title(files[i])
    ax.set_extent(Extents[i],XYproj)
#    ax.add_feature(cfeature.OCEAN,edgecolor=None,zorder=1,color="lightblue")
#    ax.add_feature(cfeature.LAKES,edgecolor=None,zorder=1,color="lightblue")
    ax.add_feature(cfeature.COASTLINE,zorder=1,edgecolor="grey")
    ax.add_feature(cfeature.STATES,zorder=1,edgecolor="gray")
    ax.scatter(Cities["Lon"],Cities["Lat"],s=Cities["City Pop"]*2e-5,marker="o",c="k",zorder=2,transform=XYproj)
#plt.close(fig)


# In[9]:


ALL_Routes, w_matrix_ALL, dr_matrix = makeRoutes(Cities,dr_matrix=dr_matrix)
#for i in range(len(ALL_Routes[1])):
#               print(i,
#                     ALL_Routes[1][i],",",
#                     ALL_Routes[2][i],",",
#                     ALL_Routes[3][i],",",
#                     ALL_Routes[4][i],",",
#                     ALL_Routes[5][i])


# In[10]:


citypairs =[["Detroit-Warren-Dearborn","Cleveland-Elyria"],
            ['Washington-Arlington-Alexandria','Atlantic City-Hammonton'],
            ["Milwaukee-Waukesha","Grand Rapids-Kentwood"],
            ["Grand Rapids-Kentwood","Appleton"],
            ["Grand Rapids-Kentwood","Madison"],
            ["Grand Rapids-Kentwood","Racine"],
            ["Janesville-Beloit","Grand Rapids-Kentwood"],
            ["Monroe","Appleton"],
            ["Monroe","Racine"],
            ["Monroe","Erie"],
            ["Cleveland-Elyria" , "Monroe"],
            ["San Francisco-Oakland-Berkeley","Vallejo"],
            ["San Francisco-Oakland-Berkeley","Napa"],
            ["San Francisco-Oakland-Berkeley","Santa Rosa-Petaluma"],
            ["Detroit-Warren-Dearborn","Erie"],
            ["Detroit-Warren-Dearborn","Buffalo-Cheektowaga"],
            ["Virginia Beach-Norfolk-Newport News","Atlantic City-Hammonton"],
            ["Baltimore-Columbia-Towson","Atlantic City-Hammonton"],
            ["Grand Rapids-Kentwood","Green Bay"],
            ["Lansing-East Lansing","Green Bay"],
            ["Green Bay","Niles"],
            ["Green Bay" , "Battle Creek"],
            ["Appleton" , "Battle Creek"],
            ["Appleton", "Saginaw"],
            ['Richmond','Atlantic City-Hammonton'],
            ['Salisbury','Winchester'],
            ["Green Bay" , "Saginaw"],
            ["Chicago-Naperville-Elgin","Muskegon"],
            ["Chicago-Naperville-Elgin","Niles"],
            ["Chicago-Naperville-Elgin","Battle Creek"],
            ["Chicago-Naperville-Elgin","Niles"],
            ["Grand Rapids-Kentwood","Chicago-Naperville-Elgin"],
            ["Chicago-Naperville-Elgin","Jackson"],
            ["Milwaukee-Waukesha","Jackson"],
            ["Sheboygan","Jackson"],
            ["Racine","Battle Creek"],
            ["Milwaukee-Waukesha","Battle Creek"],
            ["Milwaukee-Waukesha","Saginaw"],
            ["Rockford","Niles"],
            ["Grand Rapids-Kentwood" , "Rockford"],
            ["Saginaw","Sheboygan"],
            ["Saginaw","Racine"],
            ["Janesville-Beloit","Niles"],
            ["Racine","Niles"],
            ["Appleton","Niles"],
            ["Chicago-Naperville-Elgin","Kalamazoo-Portage"],
            ["Oshkosh-Neenah","Kalamazoo-Portage"],
            ["Rockford","Kalamazoo-Portage"],
            ["Janesville-Beloit","Kalamazoo-Portage"],
            ["Kalamazoo-Portage","Racine"],
            ["Racine","Jackson"],
            ["Kalamazoo-Portage","Sheboygan"],
            ["Green Bay","Kalamazoo-Portage"],
            ["Niles","Sheboygan"],
            ["Milwaukee-Waukesha","Kalamazoo-Portage"],
            ['Duluth','Muskegon'],
            ["Milwaukee-Waukesha","Niles"],
            ["Green Bay" , "Muskegon"],
            ["Milwaukee-Waukesha" , "Muskegon"],
            ["Appleton" , "Muskegon"],
            ["Kalamazoo-Portage","Appleton"],
            ["Rockford","Muskegon"],
            ["Muskegon","Racine"],
            ["Lansing-East Lansing" , "Sheboygan"],
            ["Lansing-East Lansing","Racine"],
            ["Milwaukee-Waukesha","Lansing-East Lansing"],
            ["Lansing-East Lansing","Appleton"],
            ["Madison","Muskegon"],
            ["Madison","Niles"],
            ["Oshkosh-Neenah","Niles"],
            ["Muskegon" , "Oshkosh-Neenah"],
            ["Grand Rapids-Kentwood","Oshkosh-Neenah"],
            ["Muskegon","Janesville-Beloit"],
            ["Muskegon" , "Wausau-Weston"],
            ["Grand Rapids-Kentwood" , "Wausau-Weston"],
            ["Seattle-Tacoma-Bellevue","Bremerton-Silverdale-Port Orchard"],
            ["Bremerton-Silverdale-Port Orchard","Bellingham"],
            ["Bremerton-Silverdale-Port Orchard","Mount Vernon-Anacortes"],
            ["Salisbury","Hagerstown-Martinsburg"],
            ["Salisbury","Goldsboro"],
            ["Salisbury","New Bern"],
            ["Durham-Chapel Hill","Salisbury"],
            ["Salisbury" , "Greenville"],
            ["Salisbury","Burlington"],
            ["Raleigh-Cary","Salisbury"],
            ["Richmond","Salisbury"],
            ["Atlantic City-Hammonton","Salisbury"],
            ["Virginia Beach-Norfolk-Newport News","Salisbury"],
            ["Washington-Arlington-Alexandria","Salisbury"],
            ["Salisbury","Charlottesville"],
            ["Durham-Chapel Hill","Salisbury"],
            ["Baltimore-Columbia-Towson","Salisbury"],
            ["Salisbury","Lynchburg"],
            ["Salisbury" , "Greenville"],
            ["Salisbury", "Jacksonville"],
            ["Durham-Chapel Hill","Vineland-Bridgeton"],
            ["Virginia Beach-Norfolk-Newport News","Vineland-Bridgeton"],
            ["Washington-Arlington-Alexandria","Vineland-Bridgeton"],
            ["Richmond","Vineland-Bridgeton"],
            ["Greenville" , "Vineland-Bridgeton"],
            ["Salisbury","Vineland-Bridgeton"],
            ["Salisbury","Rocky Mount"],
            ["Vineland-Bridgeton","Rocky Mount"],
            ["Charlottesville","Vineland-Bridgeton"],
            ["Greenville" , "Dover"],
            ["Dover","Charlottesville"],
            ["Atlantic City-Hammonton","Dover"],
            ["Virginia Beach-Norfolk-Newport News","Dover"],
            ["Washington-Arlington-Alexandria","Dover"],
            ["Baltimore-Columbia-Towson","Dover"],
            ["Richmond","Dover"],
            ["Panama City","Homosassa Springs"],
            ["Baltimore-Columbia-Towson","Vineland-Bridgeton"],
            ["Dover","Rocky Mount"],
            ["Lynchburg","Dover"],
            ["Durham-Chapel Hill","Dover"],
            ["Raleigh-Cary","Dover"],
            ["Dover","Burlington"],
            ["Greensboro-High Point","Dover"],
            ["Dover","Vineland-Bridgeton"],
            ["Vineland-Bridgeton","California-Lexington Park"],
            ["Dover","California-Lexington Park"],
            ["Salisbury","California-Lexington Park"],
            ["Richmond","California-Lexington Park"],
            ["Vineland-Bridgeton","California-Lexington Park"],
            ["Muskegon","Sheboygan"],
            ["Muskegon","La Crosse-Onalaska"],
            ["Muskegon","Eau Claire"],
            ["Cedar Rapids","Muskegon"],
            ["Grand Rapids-Kentwood","Sheboygan"],
            ["Battle Creek","Sheboygan"],
            ["Dover" , "New Bern"],
            ["Rocky Mount","California-Lexington Park"],
            ["Greenville","California-Lexington Park"],
            ["Virginia Beach-Norfolk-Newport News","California-Lexington Park"],
            ["Tampa-St Petersburg-Clearwater" , "Panama City"],
            ["Stockton","Vallejo"],
            ["San Jose-Sunnyvale-Santa Clara","Vallejo"],
            ["Modesto","Vallejo"],
            ["Vallejo","Santa Cruz-Watsonville"],
            ["Vallejo","Salinas"],
            ["Stockton","Napa"],
            ["San Jose-Sunnyvale-Santa Clara","Napa"],
            ["San Jose-Sunnyvale-Santa Clara","Santa Rosa-Petaluma"],
            ["Stockton","Santa Rosa-Petaluma"],
            ["Santa Cruz-Watsonville","Napa"],
            ["Santa Rosa-Petaluma","Santa Cruz-Watsonville"],
            ["Bremerton-Silverdale-Port Orchard","Wenatchee"],
            ["Salisbury","Jacksonville"],
            ['Atlantic City-Hammonton', 'California-Lexington Park'],
            ['Detroit-Warren-Dearborn','Rochester'],
            ['Dover','Winchester'],
            ['Lynchburg','Vineland-Bridgeton'],
            ['Raleigh-Cary','Vineland-Bridgeton'],
            ['Rochester','Muskegon'],
            ['Minneapolis-St Paul-Bloomington','Muskegon'],
            ['Atlantic City-Hammonton','Rocky Mount'],
            ['Vineland-Bridgeton','New Bern'],
            ['Atlantic City-Hammonton','Greenville'],
            ['Vineland-Bridgeton','Goldsboro'],
            ['Salisbury','Harrisonburg'],
            ['Atlantic City-Hammonton','New Bern'],
            ['Salisbury','Staunton'],
            ['Salisbury','Roanoke'],
            ['Atlantic City-Hammonton','Goldsboro'],
            ['Jacksonville','Vineland-Bridgeton'],
            ['Atlantic City-Hammonton','Charlottesville'],
            ['Burlington','Vineland-Bridgeton'],
            ['Durham-Chapel Hill','Atlantic City-Hammonton'],
            ['Greensboro-High Point','Salisbury'],
            ['Salisbury','Blacksburg-Christiansburg'],
            ['Fayetteville','Salisbury'],
            ['Winston-Salem','Salisbury'],
            ['Dover','Goldsboro'],
            ['Jacksonville','Dover'],
            ['Salisbury','Wilmington'],
            ['Salisbury','Beckley'],
            ['Charlotte-Concord-Gastonia','Salisbury'],
            ['Salisbury','Florence'],
            ['Durham-Chapel Hill','California-Lexington Park'],
            ['Raleigh-Cary','California-Lexington Park'],
            ['Goldsboro','California-Lexington Park'],
            ['New Bern','California-Lexington Park']
#           ["San Diego-Chula Vista-Carlsbad","El Centro"],#
#           ["San Diego-Chula Vista-Carlsbad","Yuma"],#
#           ["Phoenix-Mesa-Chandler","San Diego-Chula Vista-Carlsbad"],#
#           ["San Diego-Chula Vista-Carlsbad","Prescott Valley-Prescott"],#
#           ["Phoenix-Mesa-Chandler","Riverside-San Bernardino-Ontario"]#
           ]
Routeids =[]
iswet = ALL_Routes["Weight"]<0
for citypair in citypairs:
    iswet = iswet|(ALL_Routes["Large City"]==citypair[0])&(ALL_Routes["Small City"]==citypair[1])
    iswet = iswet|(ALL_Routes["Large City"]==citypair[1])&(ALL_Routes["Small City"]==citypair[0])
Routeids = ALL_Routes.index[iswet].values
#for i,row in ALL_Routes[iswet].iterrows():
#    print(row.values)
print(w_matrix_ALL.shape,len(ALL_Routes.index),"-",len(Routeids))
Dry_Routes, Dry_weights, dr_matrix = RemoveRoutes(Cities,
                                       w_matrix=w_matrix_ALL,
                                       dr_matrix=dr_matrix,
                                       Routes=ALL_Routes,
                                       Routeids=Routeids,
                                       verbose=False)
print(w_matrix_ALL.shape,len(ALL_Routes.index))
print(Dry_weights.shape,len(Dry_Routes.index))


# In[11]:


dr0 = 4
Trimmed_Routes, Trimmed_weights, dr_matrix = TrimRoutes(Cities,
                                             Routes=Dry_Routes,
                                             weights=Dry_weights,
                                             dr_matrix=dr_matrix,
                                             dr0=dr0,
                                             useweights=2,
                                             very_verbose=False)
print(w_matrix_ALL.shape,len(ALL_Routes.index))
print(Dry_weights.shape,len(Dry_Routes.index))
print(Trimmed_weights.shape,len(Trimmed_Routes.index))
Wmax = 80000
fig = plt.figure()
ax = fig.add_axes((.05,.0,.95,1.0), projection=proj)
xycords = ccrs.PlateCarree()._as_mpl_transform(ax)
ax.add_feature(cfeature.COASTLINE,linewidth=1,zorder=2,edgecolor="darkgrey")
ax.add_feature(cfeature.STATES,linewidth=1,zorder=2,edgecolor="grey")
lonmin = -135
lonmax = -60
latmin = 30
latmax = 50
ax.set_extent([lonmin,lonmax,latmin,latmax],XYproj)
for i,row in Trimmed_Routes[(Trimmed_Routes["Weight"]<Wmax)&(Trimmed_Routes["Weight"]>0.0*Wmax)].iterrows():
#for i,row in ALL_Routes[iswet].iterrows():
    lons = Cities["Lon"][[row["Large Index"],row["Small Index"]]]
    lats = Cities["Lat"][[row["Large Index"],row["Small Index"]]]
    if np.min(lons)>lonmin and np.max(lons) < lonmax and np.min(lats) > latmin and np.max(lats) < latmax:
        ax.plot(lons,lats,transform=XYproj)
        for idxsize in ["Large Index","Small Index"]:
            citytext = Cities["Cities"][row[idxsize]].split("-")[0].split("/")[0]
            citytext = citytext.replace(" ","\n")
            cityxy = (Cities["Lon"][row[idxsize]],
                   Cities["Lat"][row[idxsize]])
            ax.annotate(xy=cityxy,
                text=citytext,
                xycoords = xycords,
                ha="center", va="center",
                rotation = 0,
                fontsize = 20,
                bbox = dict(boxstyle="round",pad=0, fc=(1,1,1,0.5),edgecolor=None,lw=0),
                annotation_clip=True,
                    zorder=5)
        print("['"+Cities["Cities"][row["Large Index"]]+"','"+Cities["Cities"][row["Small Index"]]+"']")
#Trimmed_Routes[(Trimmed_Routes["Weight"]<Wmax)&(Trimmed_Routes["Weight"]>.5*Wmax)]


# In[12]:


#print(Trimmed_Routes[0])
dq=.0625
for q in np.arange(0,1+dq,dq):
    print(
        np.floor(np.quantile(Dry_Routes["Weight"],q)),
        np.floor(np.quantile(Trimmed_Routes["Weight"],q)),
        str(q*100)+"%")


# In[13]:


#weightlimit = np.quantile(Trimmed_Routes[0],.000)
weightlimit = 0.0
#weightlimit = 5000.0
if weightlimit>np.min(Trimmed_Routes["Weight"]):
    Routeids = Trimmed_Routes[Trimmed_Routes["Weight"]<weightlimit].index.values
    print(Routeids)
    Routes, weights, dr_matrix = RemoveRoutes(Cities,
                                       w_matrix=weights,
                                       dr_matrix=dr_matrix,
                                       Routes=Trimmed_Routes,
                                       Routeids=Routeids,
                                       verbose=False)
else:
    Routeids = []
    weights = Trimmed_weights*1
    Routes = Trimmed_Routes
print(w_matrix_ALL.shape,len(ALL_Routes.index))
print(Dry_weights.shape,len(Dry_Routes.index))
print(Trimmed_weights.shape,len(Trimmed_Routes.index))
print(len(Trimmed_Routes.index),"-",len(Routeids))
print(weights.shape,len(Routes.index))


# In[19]:


plt.close('all')
Connections = np.sum(np.sign(weights), axis = 1)
maxweight = np.max(Routes["Weight"])
minweight = np.min(Routes["Weight"])
cmap = cm.get_cmap('tab20b') #routes
gamma = 1/1.5
smap = cm.get_cmap('autumn') #cities
zmap = cm.get_cmap('binary') #Elevation
ymap = cm.get_cmap('YlGn_r') # Alt Elevation
N_cities = len(Cities)
maxpop = np.nanmax(Cities["City Pop"][Connections>0])
minpop = np.nanmin(Cities["City Pop"][Connections>0])
minprom = np.nanmin(Cities["Prominence"][Connections>0])
#### This makes the figure
# Each subplot gets its own set of axes
fig = plt.figure()
#proj = ccrs.Orthographic(
#        central_longitude=np.mean(CONUS_Extent[0:2]), 
#        central_latitude=np.mean(CONUS_Extent[2:4]), 
#        globe=None)
ax = fig.add_axes((.05,.0,.95,1.0), projection=proj)
xycords = ccrs.PlateCarree()._as_mpl_transform(ax)
ax.margins(tight=True)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off#
ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off#

cax = fig.add_axes((.025,.025,.025,.95))
# put map features in the background of each plot
XX,YY = np.meshgrid(X,Y)
ZZ = 0*XX
ZZ = np.maximum(estd,ZZ)
ZZ = np.maximum(slope,ZZ)
#ZZ = np.maximum(elev,ZZ)
zlevels = np.geomspace(2**(-6),1,7)
ax.contourf(XX,YY,ZZ,transform=XYproj,zorder=0,cmap=zmap,transform_first=True,levels=zlevels)
#ax.contour(XX,YY,ZZ,transform=XYproj,zorder=0,cmap=ymap,transform_first=True,levels=zlevels)
#    ax.add_feature(cfeature.OCEAN,edgecolor=None,zorder=1,color="lightblue")
#    ax.add_feature(cfeature.LAND,edgecolor=None,zorder=0,color="white")
#    ax.add_feature(cfeature.LAKES,edgecolor=None,zorder=1,color="lightblue")
ax.add_feature(cfeature.COASTLINE,linewidth=1,zorder=2,edgecolor="darkgrey")
ax.add_feature(cfeature.STATES,linewidth=1,zorder=2,edgecolor="grey")    
#
for i in range(N_cities):
    if(Connections[i]>0):
        #City_size = np.sqrt(Cities.iloc[i]["City Pop"]/maxpop)
        City_size = np.log(Cities.iloc[i]["City Pop"]/minpop)/np.log(maxpop/minpop)
        City_prominence = Cities.iloc[i]["Prominence"]
        City_marker_size = 2+np.sqrt(City_size)*128
        Pop_percentile = Cities.iloc[i]["Pop percentile"]
        #color_value = Connections[i]/Connections.max()
        #color_value = (np.log(Cities.iloc[i]["City Pop"]/minpop))/(np.log(maxpop/minpop))
        color_value = Pop_percentile
        ax.scatter(Cities.iloc[i]["Lon"],Cities.iloc[i]["Lat"],
                       s=City_marker_size,
                       marker="o",
                       color=smap(color_value),
                       edgecolor="k",
                       #alpha=.625,
                       transform=XYproj,
                       zorder=4)
        cityfont = np.max([City_size*14,10])
        if City_prominence>90:
                citytext = Cities.iloc[i]["Cities"].split("-")[0].split("/")[0]
                citytext = citytext.replace(" ","\n")
                cityxy = (Cities.iloc[i]["Lon"], Cities.iloc[i]["Lat"])
                ax.annotate(xy=cityxy,
                    text=citytext,
                    xycoords = xycords,
                    ha="center", va="center",
                    rotation = 0,
                    fontsize = cityfont,
                    bbox = dict(boxstyle="round",pad=0, fc=(1,1,1,0.5),edgecolor=None,lw=0),
                    annotation_clip=True,
                    zorder=5)
for idx, route in Routes.iterrows():
        weight = route[0]
        if (weight>0) :
            k = route[3]
            l = route[4]
            locs = np.vstack((Cities.iloc[k][["Lon","Lat"]].values,Cities.iloc[l][["Lon","Lat"]].values))
            lw = 8
            # Draw the lines on all the subplots
#            color_value = np.log(weight/minweight)/np.log(maxweight/minweight)
#            color_value = (weight)/(maxweight)
            color_value = (weight)/(maxweight)
            color_value = np.power(color_value,gamma)
            ax.plot(
                    locs[:,0],
                    locs[:,1],
                    c=cmap(color_value),
                    lw = lw,
                    solid_capstyle='round',
                    zorder=3,
#                    alpha = alpha,
                    transform=GCproj)

cx = [0,.5,1]
cy = np.arange(0,maxweight,64)
cxx,cyy = np.meshgrid(cx,cy)
#czz = np.log(cyy/minweight)/np.log(maxweight/minweight)
#cz = np.log(cy/minweight)/np.log(maxweight/minweight)
cz = (cy)/(maxweight)
czz = (cyy)/(maxweight)
cz = np.power(cz,gamma)
czz = np.power(czz,gamma)
cax.contourf(cxx,cyy,czz,cmap=cmap,levels=cz)
#cbarpower = np.floor(np.log(maxweight)/np.log(10))
#cbarmax = np.ceil(maxweight/10**(cbarpower-3))*(10**(cbarpower-3))
cax.set_ylim(0,maxweight)
cax.set_ylabel("Route weight\n[People/km]")
cax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

for Extent, file in zip(Extents,files):
#for Extent, file in zip([Extents[0]],[files[0]]):
    print("Creating "+file)
    ax.set_extent(Extent,XYproj)
    print("Saving",file)
    fig.savefig(file)
    print("Saved "+file)
print("Done")
plt.close(fig)


# In[ ]:





# In[15]:


#Trimmed_Routes[(Trimmed_Routes["Large City"]=="Salisbury")|(Trimmed_Routes["Small City"]=="Salisbury")]
#Dry_Routes[((Dry_Routes["Large City"]=="Salisbury")|(Dry_Routes["Small City"]=="Salisbury"))&(Dry_Routes["Distance"]>1000)]
#ALL_Routes[(ALL_Routes["Large City"]=="Salisbury")|(ALL_Routes["Small City"]=="Salisbury")]
Wmax = 40000
Trimmed_Routes[(Trimmed_Routes["Weight"]<Wmax)&(Trimmed_Routes["Weight"]>.5*Wmax)]


# In[18]:


plt.semilogy(np.diff(sorted(Trimmed_Routes["Weight"])/np.median(Trimmed_Routes["Weight"])))


# In[ ]:




