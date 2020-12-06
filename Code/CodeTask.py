#!/usr/bin/env python
# coding: utf-8

# # Estimation of Energy Consumption 

# ### Table Of Contents
# * [Chapter 1: Code Task](#chapter1)
#     * [Section 1.1: Importing required libraries](#section_1_1)
#     * [Section 1.2: Loading input data csv file and importing it into dataframe](#section_1_2)
#     * [Section 1.3: Missing Data Imputation](#section_1_3)
#         * [Section 1.3.1: Finding missing values](#section_1_3_1)
#         * [Section 1.3.2: Initializing imputation using mean imputation method](#section_1_3_2)
#         * [Section 1.3.3: Imputating missing data using missForest algorithm](#section_1_3_3)
#     * [Section 1.4: Univariate Outlier Detection](#section_1_4)
#         * [Section 1.4.1: Visualizing outliers using box plot](#section_1_4_1)
#         * [Section 1.4.2: Detecting outiers using IQR outlier detection](#section_1_4_2) 
#         * [Section 1.4.3: Dropping outliers](#section_1_4_3)   
#     * [Section 1.5: Energy Consumption Estimation](#section_1_5)
#         * [Section 1.5.1: Calculating water consumption](#section_1_5_1)   
#         * [Section 1.5.2: Calculating energy consumption](#section_1_5_2) 
#     * [Section 1.6: Saving final dataframe in csv format in the same folder as input data](#section_1_6)
# * [Chapter 2: Discussion Task](#chapter2)
# 
#     
#     

# ### Chapter 1: Code Task

# #### Section 1.1: Importing required libraries

# In[1]:


# Importing the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesRegressor
import seaborn as sns #visualisation
sns.set(color_codes=True)


# #### Section 1.2: Loading input data csv file and importing it into dataframe

# In[2]:


# loading the input data into dataframe
inputFile = "C:/Users/Acer/Desktop/Devosmita/input.csv"
df = pd.read_csv(inputFile,sep="\t")
# Displaying the top 5 rows
print(df.head())
# Displaying the datatypes
print(df.dtypes)


# #### Section 1.3: Missing Data Imputation

# ##### Section 1.3.1: Finding missing values

# In[3]:


# Finding the null values
print(df.isnull().sum())


# ##### Section 1.3.2: Initializing imputation using mean imputation method

# In[4]:


# Initializing imputation using mean imputation method
imputedData = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
imputedData = pd.DataFrame(imputedData.fit_transform(df[['returntemp','supplytemp','water_state']]), columns = ['returntemp','supplytemp','water_state'])
print(imputedData)


# ##### Section 1.3.3 Imputing missing values using missForest algorithm

# In[5]:


# Imputing missing values using missForest algorithm (ExtraTreesRegressor is similiar to missforest algorithm in R)
seed = 0
# Number of trees in the forest is 10
imputer = ExtraTreesRegressor(n_estimators=10, random_state=seed)

for x in ['returntemp','supplytemp','water_state']:
    X = imputedData.loc[:, imputedData.columns != x].values
    y = imputedData[[x]].values
    model = imputer.fit(X,y)
    imputedData[x] = model.predict(X)

print(imputedData)


# #### Section 1.4: Univariate Outlier Detection

# ##### Section 1.4.1: Visualizing outliers using box plot

# In[6]:


# Plot boxplot of 'returntemp' feature
sns.boxplot(x=imputedData['returntemp'])


# In[7]:


# Plot boxplot of 'supplytemp' feature
sns.boxplot(x=imputedData['supplytemp'])


# In[8]:


# Plot boxplot of 'water_state' feature
sns.boxplot(x=imputedData['water_state'])


# ##### Section 1.4.2: Detecting outiers using IQR outlier detection

# In[9]:


# IQR outlier detection
data = imputedData['water_state']
q1,q3 = np.percentile(data,[25,75])
interQuartileRange = q3 - q1
lowerBound = q1 - (interQuartileRange*1.5)
upperBound = q3 + (interQuartileRange*1.5)
outlierIndices = np.where((data > upperBound)|(data < lowerBound))
print(outlierIndices)


# ##### Section 1.4.3: Dropping outliers

# In[10]:


# Dropping outliers from imputed data
imputedData2 = imputedData.drop([  3,  14,  29,  42,  44,  46,  58,  77,  82, 123, 131])
print(imputedData2)
# Reset dataset
imputedData2_reset = imputedData2.reset_index(drop=True)
print(imputedData2_reset)


# In[12]:


# Dropping outliers from datetime
dateTime = df[['datetime']].drop([  3,  14,  29,  42,  44,  46,  58,  77,  82, 123, 131])
print(dateTime)
# Reset datetime
dateTime_reset = dateTime.reset_index(drop=True)
print(dateTime_reset)


# In[13]:


# Concatenate datetime and imputed data
imputedData2_reset = pd.concat([dateTime_reset, imputedData2_reset], axis=1)
print(imputedData2_reset)


# #### Section 1.5: Energy Consumption Estimation
# 

# ##### Section 1.5.1: Calculating water consumption
# 
# 

# In[14]:


# Calculating water consumption
rowSize = imputedData2_reset.shape[0]
waterConsumption = [0] * rowSize
water_state = imputedData2_reset['water_state'].values
for i in range(0, rowSize-1):
    waterConsumption[i] = water_state[i+1] - water_state[i]
imputedData2_reset['water_consumption'] = pd.DataFrame(waterConsumption)
print(imputedData2_reset)


# In[15]:


# Checking that there is no such thing as negative consumption
negativeWaterConsumptionIndices = np.where(imputedData2_reset['water_consumption']<0)
print(negativeWaterConsumptionIndices)


# ##### Section 1.5.2: Calculating energy consumption
# 
# 

# In[16]:


C = 1.16
energyConsumption= [0] * 157
waterConsumption = imputedData2_reset['water_consumption'].values
supplyTemp = imputedData2_reset['supplytemp'].values
returnTemp = imputedData2_reset['returntemp'].values
for i in range(0, 156):
    energyConsumption[i] = waterConsumption[i] * (supplyTemp[i] - returnTemp[i]) * C
imputedData2_reset['energy_consumption'] = pd.DataFrame(energyConsumption)
print(imputedData2_reset)


# #### Section 1.6: Saving final dataframe in csv format in the same folder as input data
# 
# 

# In[17]:


import os
path = os.path.dirname(inputFile)
imputedData2_reset.to_csv(path+'/output.csv')


# ### Chapter 2: Discussion Task

# In[18]:


# Task 2
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 6))
fig = plt.plot(imputedData2_reset['datetime'], imputedData2_reset['energy_consumption'])
plt.xticks([])
plt.xlabel('datetime')
plt.ylabel('energy_consumption')
plt.savefig(path+'/timeseries_plot.png')
plt.show()

