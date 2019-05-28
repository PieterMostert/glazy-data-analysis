import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# To be run after glazy_snapshot_processing.py

oxides = ["SiO2", "Al2O3", "B2O3", "Li2O", "K2O", "Na2O", "KNaO", "BeO", "MgO", "CaO", "SrO", "BaO", "ZnO", \
          "PbO", "P2O5", "F", "V2O5", "Cr2O3", "MnO", "MnO2", "FeO", "Fe2O3", "CoO", "NiO", "CuO", "Cu2O", \
          "CdO", "TiO2", "ZrO", "ZrO2", "SnO2"]

with open('glazes.pickle', 'rb') as f:
    data = pickle.load(f)

# Select glazes that don't contain any of the following oxides:
rejectOxides = ["BeO", "PbO", "F", "V2O5", "CdO"]
rejectOxidesBool = pd.DataFrame({ox: (data[ox+"_percent"]>0) for ox in rejectOxides})
data = data[rejectOxidesBool.sum(axis=1)==0]  

# Drop columns corresponding to oxides above
data.drop(columns=[ox+"_percent" for ox in rejectOxides], inplace=True)
data.drop(columns=[ox+"_percent_mol" for ox in rejectOxides], inplace=True)
oxides = [ox for ox in oxides if ox not in rejectOxides] #list(set(oxides)-set(rejectOxides))

# Remove crawl and crater glazes
craterBool = data["material_type_id"].isin([1150, 1160])
data = data[~craterBool]

# Remove Egyptian paste
EgyptBool = (data["name"].str[:8] == "Egyptian")
data = data[~EgyptBool]  

# Remove Davis Reds
DavisRedBool = (data["name"].str[:9] == "Davis Red")
data = data[~DavisRedBool]

# Remove Steve Davis' N series (since it contains a fair amount of wood ash) 
DavisNBool = (data["name"].str[:2] == "N ")
data = data[~DavisNBool]

# Remove some low-fire sculpture glazes
NB1Bool = (data["name"].str[:3] == "NB1")
data = data[~NB1Bool]

# Get rid of the following hand-picked glazes:
data.drop([2622, # Leach 4321 at cone 6
            1769, # Casting slip
            1750, # Same recipe (24750) classified as slip/engobe
            2650, 2177, 2161, 955, # Engobes
            13132, # More of a body than a glaze
            2338, # Notes indicate cone 6 firing, but similar to cone 06 and 04 glazes
            23494, # Cone 6-7 but similar to 04 Majolicas 'This glaze makes stable glazes run and flow'
            1394, # Contains about 30% tin oxide
            1972, # "Dry suede sculpture finish."
            6525, # Leach 4321 with Soda Feldspar, but given as cone 6-8, with photos of cone 9 firings
            2843, # "Looks and feels underfired" 
            2828, # The same as 2843
            2562, # 'Significant Flaw: underfired'
            19987, # 'I think it's underfired'
            10994, # 'Underfired at cone 6'
            7135, # 'Underfired at cone 5'
            2395, # 'A very, very dry matt pale yellow that clung very closely to the test bowl, like a slip, 
                 # Looks like an underfired surface, but very smooth.'
            2543, # 'Glaze looks underfired'
            1248, # 'I got this copy of Randys Green at a workshop, where it was described
                  # as a C 10 glaze'
            18790, # Picture looks a bit underfired
            12942, # Picture looks underfired
            16959, # Picture looks completely underfired
            23944, # PSM2. Could be fired higher
            17637, # Cone 6 with high Al2O3, SiO2, and no boron. Picture looks underfired.
            378, # 'It has not moved or melted at all'
            2759, # 'remained very high on the test tile, as to say, melted but not mature' Note that this 
                 # has similar oxides to 2689 and 605
            19428, # 'Looks under-matured :('
            1617, # '...some decorative pinholes not fully mature...'
            2433, # 'Alisa Clausen: used local red clay for Barnard. Mat, oversaturated, possibly underfired,'
            2436, # 'Not well melted on thinner areas, with small bubbles'
            3818, # Underfired?
            2464, # Underfired?
            12018, # Underfired?
            8401, # 'This variation has not been tested yet.'
            17469, # 'Untested yet.'
            1203, # 'Untested.'
            25728, # 'Test with colemanite...never tried.'
            1493, # designed to be used in combination with a slip
            25657, # 'Use this glaze as an accent glaze over Coleman's Yellow Crystal.'
            2480, 1342, # Lava glazes
            3422, # 'Jan's Crater C'. Has been unpublished, but I assume it's a crater glaze
            6064, # Crater glaze
            26386, # Crater glaze
            2876, # Specialty glaze, consisting of just Plastic Vitrox plus copper carb
            4624, # Crawl glaze
            1719, # Bead glaze
            1740, 1741, # Lichen
            1745, # Similar to 4095 (NB1+copper) and 4759 (Lithium slip)
            4712, 4800, # Textured glazes
            4773, 2324, # Contain silver nitrate
            1955, #1992, # Contain bismuth subnitrate
            25846, 12952, # Contain Forshammer Feldspar, which is missing its analysis
            5357, # I think the addition of GB is a mistake. Glaze without GB appears in Clay/Wood/Fire/Salt booklet (as cone 10 glaze)
            2110, # Cone 03, but with only 0.05 B2O3 and 0.05 R2O
            6182, # Cone 13, but same recipe also listed twice as cone 6
            2565, # Tested with Oldenwalder, but recipe uses Barnard
            1283, # high silica and quite similar to cone 10 glaze 6375, described as "Shiny, clear white (on white clay) no craze, very good."
            1266, # Dry glaze, contains Praeseodymium oxide
            7970 # GB given as 3.16%, when it should be 31.6%
            ], inplace=True)
# Edited cones in GlazyRecipes.csv for the following:
# 13/1/2019: 4803, 20661, 17685, 23782
# 30/1/2019: Brian Kemp's glazes to 6-7 instead of 6-6
# 26/4/2019: 1096 to 6-9
# 13/5/2019: 19139 to 6-6
# 4/5/2019: 2257,"Yellow Frosty" to 03 - 03. "Tends to be dry unless fired to hot 04"
# Also drop: 2473, 2466, 2459, 21687, 2348, 7809, 8111, 17053, 27412, 
#data = data.loc[~data["id"].isin([2622, 1769, 1750, 2177, 24689, 3818, 4803, 17685, 2338, 22246, 20661, 23782, 2650, \
#                                  1394, 1972, 24045, 24040, 9324])]  
# Check 4806, 1973
# 3447 reactive slip under Jan's crater glaze.
# 2047 matte frosty (underfired?)
# 4695 cone 4 instead of 04?
# 806
# 1482 Has this been tested?
# Clara's Harris Currie grid?
# 88 Underfired?

# Get rid of glazes with very large or small SiO2:Al2O3 ratios (Doesn't seem to make much difference)
data = data.loc[data["SiO2_Al2O3_ratio_umf"].between(4,20)] 

# Get rid of glazes with large temp ranges
data = data[data["upper_temp"] - data["lower_temp"] <= 60] 

# Recalculate umf using a possibly different set of fluxes
fluxes = ['Li2O', 'K2O', 'Na2O', 'MgO', 'CaO', 'SrO', 'BaO', 'ZnO', 'Fe2O3', 'CoO', 'CuO', "MnO", "MnO2", "FeO", "NiO", "Cu2O"]
#fluxes = ['Li2O', 'K2O', 'Na2O', 'MgO', 'CaO', 'SrO', 'BaO', 'ZnO', 'Fe2O3', 'CoO', 'CuO']
flux_sum = data.loc[:,[ox+"_percent_mol" for ox in fluxes]].sum(axis=1)

# Remove glazes with SiO2 below 1.8 umf:
SiO2Bool = data["SiO2_percent_mol"] > 1.8*flux_sum
data = data[SiO2Bool]
flux_sum = flux_sum[SiO2Bool]

Y = pd.Series([(data.loc[i,"lower_temp"] + data.loc[i,"upper_temp"])/2 for i in data.index], index=data.index, name="avg_temp")

# Get rid of low temp glazes
lower_bound = 1050
data = data[Y >=lower_bound]
Y = Y[Y >lower_bound] 

n, bins, patches = plt.hist(Y.to_list(),bins=35)
plt.xlabel(u'Firing temperature (℃)')
plt.ylabel("Number of recipes")
plt.show()

X_percent_mol = data.loc[:,[ox+"_percent_mol" for ox in oxides if ox not in ['K2O', 'Na2O']]]
dists = pairwise_distances(X_percent_mol)
dat = (data, X_percent_mol, Y, dists)
print('Number of recipes:', data.shape[0])
with open('glazes_cleaned.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dat, f, pickle.HIGHEST_PROTOCOL)
