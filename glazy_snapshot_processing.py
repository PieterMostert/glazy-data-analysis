import json
import pandas as pd 
import numpy as np
import pickle

with open("OrtonTemps.json", encoding='utf-8') as coneData:
    coneTemps = json.load(coneData)["coneTemps"]  # Temperature in C that brings down a regular self-supporting cone 
                                                  # if rate of temperature rise is 60C per hour for the last 100C

# read columns containing: id, name, material_type_id, from_orton_cone, to_orton_cone, is_analysis, is_primitive, 
# "SiO2_percent", ..., "Lu2O3_percent", "SiO2_percent_mol", ..., "Lu2O3_percent_mol", "SiO2_Al2O3_ratio_umf", "R2O_umf", "RO_umf"
data = pd.read_csv("GlazyRecipes.csv", usecols=[0,1,3,10,11,12,13]+list(range(16,77))+list(range(199,263)))
data.set_index("id", inplace=True)
boolSelect = ~(data["is_analysis"]==1) & ~(data["is_primitive"]==1) & (data["material_type_id"].between(460,1170))
data = data.loc[boolSelect]
data.drop(columns=["is_analysis","is_primitive"], inplace=True)
# data now consists of only glaze recipes. We still need to do some more data cleaning.
oxides = ["SiO2", "Al2O3", "B2O3", "Li2O", "K2O", "Na2O", "KNaO", "BeO", "MgO", "CaO", "SrO", "BaO", "ZnO", \
          "PbO", "P2O5", "F", "V2O5", "Cr2O3", "MnO", "MnO2", "FeO", "Fe2O3", "CoO", "NiO", "CuO", "Cu2O", \
          "CdO", "TiO2", "ZrO", "ZrO2", "SnO2"]
n1 = data.shape[0]
data = data.drop_duplicates(subset=[ox+"_percent_mol" for ox in oxides])  # Should take into account duplicate glazes with \
n2 = data.shape[0]                                                                      # different cone ranges
print("Number of duplicates dropped:", n1 - n2)

rareEarths = ['HfO2', 'Nb2O5', 'Ta2O5', 'MoO3', 'WO3', 'OsO2', 'IrO2', 'PtO2', 'Ag2O', 'Au2O3', 'GeO2', 'As2O3', \
              'Sb2O3', 'Bi2O3', 'SeO2', 'La2O3', 'CeO2', 'PrO2', 'Pr2O3', 'Nd2O3', 'U3O8', 'Sm2O3', 'Eu2O3', 'Tb2O3',\
               'Dy2O3', 'Ho2O3', 'Er2O3', 'Tm2O3', 'Yb2O3', 'Lu2O3']
# First, remove any recipes containing rare earth metals
rareEarthBool = pd.DataFrame({re: (data[re+"_percent"]>0) for re in rareEarths})
data = data.loc[rareEarthBool.sum(axis=1)==0]  # Select glazes that don't contain any rare earth metals

# Drop columns corresponding to rare earths
data.drop(columns=[re+"_percent" for re in rareEarths], inplace=True)
data.drop(columns=[re+"_percent_mol" for re in rareEarths], inplace=True)

oxides = ["SiO2", "Al2O3", "B2O3", "Li2O", "K2O", "Na2O", "KNaO", "BeO", "MgO", "CaO", "SrO", "BaO", "ZnO", \
          "PbO", "P2O5", "F", "V2O5", "Cr2O3", "MnO", "MnO2", "FeO", "Fe2O3", "CoO", "NiO", "CuO", "Cu2O", \
          "CdO", "TiO2", "ZrO", "ZrO2", "SnO2"]

# Remove any recipes with no oxides
oxidesBool = pd.DataFrame({ox: (data[ox+"_percent"]>0) for ox in oxides})
data = data.loc[oxidesBool.sum(axis=1)>0]

# Get rid of glazes with neither lower nor upper cones.
existing_lower_cones = ~data["from_orton_cone"].isin([np.nan])
existing_upper_cones = ~data["to_orton_cone"].isin([np.nan])
data = data.loc[existing_lower_cones | existing_upper_cones]

# Convert all cones to strings
data.loc[:, "from_orton_cone": "to_orton_cone"] = data.loc[:, "from_orton_cone": "to_orton_cone"].applymap(str)

# If "from_orton_cone" or "to_orton_cone" is missing, replace with "to_orton_cone" or "from_orton_cone" respectively
data["from_orton_cone"] = data["from_orton_cone"]*existing_lower_cones.astype('int') +  data["to_orton_cone"]*(~existing_lower_cones).astype('int')
data["to_orton_cone"] = data["from_orton_cone"]*(~existing_upper_cones).astype('int') +  data["to_orton_cone"]*existing_upper_cones.astype('int')

# Add column of temperatures
tempcols = data.loc[:, "from_orton_cone": "to_orton_cone"].applymap(lambda x: coneTemps[x])
tempcols.rename(columns={"from_orton_cone": "lower_temp", "to_orton_cone":"upper_temp"}, inplace=True)
data = pd.concat([data, tempcols], axis=1)

with open('glazes.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
