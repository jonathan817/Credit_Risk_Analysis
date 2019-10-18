# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:48:55 2019

@author: jonat
"""

import numpy as np
import pandas as pd
import scipy
from math import sqrt
import matplotlib.pyplot as plt
import os 


# Estimadors.

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import Linear_model


# Model metrics.

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# Cross Validation.

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# Importing the data.

os.chdir('C:/Users/jonat/Documents/UBIQUM/GITHUB PROJECTS/Credit_Risk_Analysis_Python/Credit_Risk_Analysis/DataSets')

rawData = pd.read_csv('default_of_credit_card_clients.csv', header=1)
rawData.head()

rawData.info()

#features

features = rawData.iloc[:,12:23]
print('Summary of feature sample')
features.head()

#dependent variable

depVar = rawData['PAY_AMT6']


#Training Set (Feature Space: X Training)

X_train = (features[: 1000])
X_train.head()


#Dependent Variable Training Set (y Training)

y_train = depVar[: 1000]
y_train_count = len(y_train.index)
print('The number of observations in the Y training set are:',str(y_train_count))
y_train.head()


#Testing Set (X Testing)

X_test = features[-100:]
X_test_count = len(X_test.index)
print('The number of observations in the feature testing set is:',str(X_test_count))
print(X_test.head())


#Ground Truth (y_test) 
y_test = depVar[:-100]
y_test_count = len(y_test.index)
print('The number of observations in the Y training set are:',str(y_test_count))
y_test.head()


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

X_train.shape, X_test.shape

model = LinearRegression(n_jobs=10)

# Models.

modelSVR = SVR()
modelRF = RandomForestRegressor()
modelLR = LinearRegression()

# Features
features = rawData.iloc[:,12:23]
print('Summary of feature sample')
features.head()

# Checking the dependent variable

print(depVar)

# Training the models.

model.fit(X_train,y_train)

# Random Forest.

modelRF.fit(X_train,y_train)


# Super Vector Regression.

modelSVR.fit(X_train,y_train)


# Linear Regression.

modelLR.fit(X_train,y_train)


from sklearn.model_selection import cross_val_score


# Scoring the models.

print(cross_val_score(modelRF, X_train, y_train)) 

print(cross_val_score(modelSVR, X_train, y_train)) 

print(cross_val_score(modelLR, X_train, y_train)) 


model.score(X_train,y_train)

modelRF.score(X_train,y_train)
              
modelSVR.score(X_train,y_train)

modelLR.score(X_train,y_train)


# Making predictions.

predictions = modelRF.predict(X_test)


# Evaluating the results.

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

rmse = sqrt(mean_squared_error(y_test, predictions))

predRsquared = r2_score(y_test,predictions)

print('R Squared: %.3f' % predRsquared)
print('RMSE: %.3f' % rmse)


plt.scatter(y_test, predictions, c=['blue','green'], alpha = 0.5)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.show();




