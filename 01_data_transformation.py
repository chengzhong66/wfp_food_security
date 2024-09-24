# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 23:24:13 2024

@author: CZhong
"""

import os
import numpy as np
import pandas as pd
import json
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point

# pth = r"C:\Users\CZhong\Desktop\data_for_test"
pth = r"Q:\DATA\SPRAIRU\CZ\Code\data_for_test"
os.chdir(pth)

food_security = pd.read_csv(os.path.join(pth, 'food_security.csv'))
geometries = gpd.read_file(os.path.join(pth, 'geometries.shp'))

conflict_path = os.path.join(pth, 'conflict.json')
economy_index_path = os.path.join(pth, 'economy_index.json')
rainfall_path = os.path.join(pth, 'rainfall.json')

with open(conflict_path) as f:
    conflict = json.load(f)

with open(economy_index_path) as f:
    economy_index = json.load(f)

with open(rainfall_path) as f:
    rainfall = json.load(f)

print(food_security.head())
print(food_security.columns)
print(geometries.head())
print(conflict.keys())
print(economy_index.keys())
print(rainfall.keys())

#%%% Transform food security to long format

df_food_security = food_security.reset_index().rename(columns={'index': 'timestamps'})
df_food_security_long = pd.melt(df_food_security, id_vars=['timestamps'], var_name='key', value_name='food_security_value')

#%%% Transform conflict json to dataframe

def process_conflict(json_data, value_column):
    # Empty list to store all expanded rows
    expanded_rows = []
    
    # Iterate over each country in the JSON data
    for country, data in json_data.items():
        # Extract the 'timestamps' and the values for the given column (conflict, economy, etc.)
        values = data[value_column]
        timestamps = data['timestamps']
        
        # Expand the rows: create a row for each timestamp
        for i in range(len(timestamps)):
            expanded_rows.append({'key': country, 'timestamps': timestamps[i], value_column: values[i]})
    
    # Convert the expanded rows into a DataFrame
    return pd.DataFrame(expanded_rows)

df_conflict_long = process_conflict(conflict, 'conflict_index')

#%%% Transform economy index json to dataframe

data = []
for key, value in economy_index.items():
    # Extract 'country_index', 'longlat', and 'timestamps' for each entry
    country_index = value.get('country_index', None)
    longlat = value.get('longlat', None)
    timestamps = value.get('timestamps', None)
    
    # Append the data as a row
    data.append({
        'key': key,
        'country_index': country_index,
        'longlat': longlat,
        'timestamps': timestamps
    })

df_economy_index = pd.DataFrame(data)
df_economy_index.head()

# Check if 'timestamps' and 'country_index' are lists, if not, wrap them in a list
df_economy_index['timestamps'] = df_economy_index['timestamps'].apply(lambda x: x if isinstance(x, list) else [x])
df_economy_index['country_index'] = df_economy_index['country_index'].apply(lambda x: x if isinstance(x, list) else [x])

def process_economy(row):
    # Ensure timestamps and country_index are the same length
    min_length = min(len(row['timestamps']), len(row['country_index']))
    
    # Truncate both lists to the same length
    timestamps = row['timestamps'][:min_length]
    country_index = row['country_index'][:min_length]
    
    # Return a DataFrame for each row
    return pd.DataFrame({
        'key': [row['key']] * min_length,
        'longlat': [row['longlat']] * min_length,
        'timestamps': timestamps,
        'country_index': country_index
    })

# Apply the process_row function row-wise to create a long-format DataFrame
df_economy_index_long = pd.concat(df_economy_index.apply(process_economy, axis=1).values).reset_index(drop=True)
df_economy_index_long.head()

#%%% Transform rainfall json to dataframe

def process_rainfall_row(key, row):
    # Ensure timestamps and rainfall_mm are the same length
    min_length = min(len(row['timestamps']), len(row['rainfall_mm']))
    
    # Truncate both lists to the same length
    timestamps = row['timestamps'][:min_length]
    rainfall_mm = row['rainfall_mm'][:min_length]
    
    # Return a DataFrame for this key
    return pd.DataFrame({
        'key': [key] * min_length,
        'geometry': [row['geometry']] * min_length,
        'timestamps': timestamps,
        'rainfall_mm': rainfall_mm
    })

df_rainfall_long = pd.concat([process_rainfall_row(key, value) for key, value in rainfall.items()], ignore_index=True)
df_rainfall_long.head()

#%%% Merge with geographical information

# First merge rainfall and geometries using geopandas
col_dict = {'key':'key_rainfall'}
df_rainfall_long.rename(columns=col_dict, inplace=True)

df_rainfall_long['geometry'] = df_rainfall_long['geometry'].apply(lambda g: wkt.loads(g))
gdf_polygons_rainfall = gpd.GeoDataFrame(df_rainfall_long, geometry='geometry')
gdf_polygons = gpd.GeoDataFrame(geometries, geometry='geometry')

gdf_polygons_rainfall = gdf_polygons_rainfall.set_crs('EPSG:4326')
gdf_polygons = gdf_polygons.set_crs('EPSG:4326')

gdf_merged_rainfall = gpd.sjoin_nearest(gdf_polygons_rainfall, gdf_polygons, how='left') #!!! #all 31 countries have rainfall_mm
gdf_merged_rainfall = gdf_merged_rainfall.drop(columns=['index_right'])
# gdf_merged_rainfall['avg_rainfall_mm'] = gdf_merged_rainfall.groupby(['name', 'timestamps'])['rainfall_mm'].transform('mean')

grouped_rainfall = gdf_merged_rainfall.groupby(['name', 'timestamps'], as_index=False)['rainfall_mm'].mean()
grouped_rainfall.rename(columns={'rainfall_mm': 'avg_rainfall_mm'}, inplace=True)

# Then merge geometries on economy index using geopandas

# Convert df_economy_index_long into gdf_points
df_economy_index_long['geometry'] = df_economy_index_long['longlat'].apply(lambda x: Point(x[0], x[1]))
gdf_points = gpd.GeoDataFrame(df_economy_index_long, geometry='geometry')

# Convert geometries to a GeoDataFrame
gdf_polygons = gpd.GeoDataFrame(geometries, geometry='geometry')

# Set the correct CRS (if your polygons are in latitude/longitude, use EPSG:4326)
gdf_points = gdf_points.set_crs('EPSG:4326')  # Assuming WGS84 latitude/longitude
gdf_polygons = gdf_polygons.set_crs('EPSG:4326')

# Merge geometries on economy index
gdf_merged_economy = gpd.sjoin(gdf_points, gdf_polygons, how='left', predicate='within')
gdf_merged_economy = gdf_merged_economy.drop(columns=['index_right'])

# Then merge with grouped_rainfall using name and timestamps
gdf_merged = gdf_merged_economy.merge(grouped_rainfall, on=['timestamps', 'name'], how='left')
gdf_merged = gdf_merged.drop(columns=['index_right'])

#%%% Merge with conflict and food security with key

gdf_merged = gdf_merged.drop(columns=['key']) #!!!check if have time, repetitive 1, 3, 9, 11, 13

df_conflict_long.rename(columns = {'key':'name'}, inplace = True) # rename to merge
gdf_merged['name'] = gdf_merged['name'].astype(str)
df_conflict_long['name'] = df_conflict_long['name'].astype(str)
gdf_merged = gdf_merged.merge(df_conflict_long, on=['timestamps', 'name'], how='left')

df_food_security_long.rename(columns = {'key':'name'}, inplace = True)
gdf_merged['name'] = gdf_merged['name'].astype(str)
df_food_security_long['name'] = df_food_security_long['name'].astype(str)
gdf_merged = gdf_merged.merge(df_food_security_long, on=['timestamps', 'name'], how='left')

#%%% Final steps

gdf_merged = gdf_merged.drop(columns=['geometry'])
gdf_merged = gdf_merged[['name','timestamps','longlat','country_index','avg_rainfall_mm','conflict_index','food_security_value']]
gdf_merged.rename(columns = {'country_index':'economy_index'}, inplace = True)
gdf_merged.describe()








