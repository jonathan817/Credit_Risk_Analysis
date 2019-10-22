# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:58:30 2019

@author: jonat
"""
# IMPORTING RELEVANT PACKAGES.

import math
import math as mt

import numpy as np
import pandas as pd
import os 
from pandas import Series, DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy

import xgboost as xgb

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

from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split

# Classifiers Models.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier 
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

# To upsample - downsample

from sklearn.utils import resample

 
# ROC Curve.

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

from sklearn.calibration import CalibratedClassifierCV,calibration_curve



# Set default matplot figure size
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


math.pi

import math as mt

mt.pi

mt.sqrt(25)

import numpy as np

from numpy import random

dataOne = random.rand(5,5)

np.mean(dataOne)

np.sqrt(dataOne)

print('I Love data Science')

vector= [1,2,3,4,5]

print(vector)

def my_function(x): return 5 * x

my_function(10)

variableName = 25

variableName1 = 'I Love data Science'

print(variableName1)

variableName = 25

type(variableName)

variableName = 25.0

varOne = 25

varTwo = 25.0

varThree = varOne + varTwo

print(varThree)


varOne = 25

varTwo = 'Hello'

varThree =  varOne+ varTwo

print(varThree)

s = "10010"

c = int(s,2) 

print ("After converting to integer base 2 : ", end="") 

print (c) 

e = float(s) 
print ("After converting to float : ", end="") 
print (e) 

listName = [ ]

listOne = [1,2,3,4]

print(listOne[0:4])

tel = {'jack': 4098, 'sape': 4139}

tel['jack']


tel = {'jack': 4098, 'sape': 4139, 'juan': 4512, 'pepe': 4987, 'jose': 4986}

tel['jose']


# ---------------------------------------------------------------------------

import pandas as pd

import os 

os.chdir('C:/Users/jonat/Documents/UBIQUM/GITHUB PROJECTS/Credit_Risk_Analysis_Python/Credit_Risk_Analysis/DataSets')

credit = pd.read_csv('default_of_credit_card_clients.csv', header =1)

credit.head()

credit.describe()

credit.info() 

import pandas as pd

import matplotlib.pyplot as plt

header = credit.dtypes.index

print(header)

plt.hist(credit['LIMIT_BAL'])

plt.show()

plt.hist(credit['SEX'])

plt.hist(credit['EDUCATION'])

plt.hist(credit['AGE'])

plt.plot(credit['LIMIT_BAL'])

plt.show()



# How to select some rows

credit.iloc[:,7:12]


# Changing the Pay Columns.

# credit.loc[credit['PAY_0'] < -2, 'PAY_0'] = -2

# credit.loc[credit['PAY_2'] < -2, 'PAY_2'] = -1

# credit.loc[credit['PAY_3'] < -2, 'PAY_3'] = -1

# credit.loc[credit['PAY_4'] < -2, 'PAY_4'] = -1

# credit.loc[credit['PAY_5'] < -2, 'PAY_5'] = -1

# credit.loc[credit['PAY_6'] < -2, 'PAY_6'] = -1

credit.shape


# credit = credit[credit.PAY_0 != 0]

# credit = credit[credit.PAY_2 != 0]

# credit = credit[credit.PAY_3 != 0]

# credit = credit[credit.PAY_4 != 0]

# credit = credit[credit.PAY_5 != 0]

# credit = credit[credit.PAY_6 != 0]

# credit_2 = credit.drop(credit[(credit['PAY_0'] == 0) & (credit['PAY_2'] == 0) & (credit['PAY_3'] == 0) & (credit['PAY_4'] == 0) & (credit['PAY_5'] == 0) & (credit['PAY_6'] == 0)].index)

# credit_2 = credit_2.drop(credit_2[(credit_2['BILL_AMT1'] == 0) & (credit_2['BILL_AMT2'] == 0) & (credit_2['BILL_AMT3'] == 0) & (credit_2['BILL_AMT4'] == 0) & (credit_2['BILL_AMT5'] == 0) & (credit_2['BILL_AMT6'] == 0)].index)

# credit_2 = credit_2.drop(credit_2[(credit_2['PAY_AMT1'] == 0) & (credit_2['PAY_AMT2'] == 0) & (credit_2['PAY_AMT3'] == 0) & (credit_2['PAY_AMT4'] == 0) & (credit_2['PAY_AMT5'] == 0) & (credit_2['PAY_AMT6'] == 0)].index)

x = credit['PAY_0']

y = credit['PAY_2']

plt.scatter(x,y)

plt.show()


# Creating a boxplot.

header = credit.dtypes.index

print(header)

A = credit['BILL_AMT1']

plt.boxplot(A,0,'gD')

plt.show()


# Creating a Correlation matrix.

corrMat = credit.corr()

print(corrMat)


# Cálculo Covarianza.

covMat = credit.cov()

print(covMat)

# Removing irrelevant data on Education (0, 5 and 6 are transform to 4) and set as an object variable (factor).

plt.hist(credit['EDUCATION'])

credit['EDUCATION'].describe()

credit.loc[credit['EDUCATION'] == 0, 'EDUCATION'] = 4

credit.loc[credit['EDUCATION'] == 5, 'EDUCATION'] = 4

credit.loc[credit['EDUCATION'] == 6, 'EDUCATION'] = 4

credit.EDUCATION.value_counts() 

# Changing some variables to Factor.

credit['EDUCATION'] = credit['EDUCATION'].astype(object)

credit['SEX'] = credit['SEX'].astype(object)

credit = credit.rename(columns = {'default payment next month':'DEFAULT'})

# Let's analyze what percentage of total transactions have been default.

Percentage_default = credit.DEFAULT.sum() / len(credit.DEFAULT)

Percentage_default

# Changing the DEFAULT colun type to factor (object)

credit['DEFAULT'] = credit['DEFAULT'].astype(object)


# Let's analyze what is the default rate between men and women.

sex_distribution = sns.factorplot('SEX', data=credit, kind='count', aspect=1.5)

default_sexo = credit.groupby(['SEX', 'DEFAULT']).size().unstack(1)

default_sexo.plot(kind='bar', stacked = True, title = 'DEFAULT BY GENDER', mark_right = True)

default_sexo_2 = default_sexo

default_sexo_2['PERCENTAGE'] = (default_sexo[1]/(default_sexo[0] + default_sexo[1])) 

default_sexo_2

# Marriage Status 3 and 0 will be combined in an unic value of 3.       

credit.loc[credit['MARRIAGE'] == 0, 'MARRIAGE'] = 3

credit['MARRIAGE'] = credit['MARRIAGE'].astype(object)

credit.MARRIAGE.value_counts() 


# Let's analyze the default rate by marital status.

default_marriage = credit.groupby(['MARRIAGE', 'DEFAULT']).size().unstack(1)

default_marriage.plot(kind='bar', stacked = True, title = 'DEFAULT BY MARITAL STAUTS', mark_right = True)

default_marriage_2 = default_marriage.copy()

default_marriage_2['PERCENTAGE'] = (default_marriage_2[1]/(default_marriage_2[0] + default_marriage_2[1])) 

default_marriage_2






# Let's analyze the default rate by education.

default_education = credit.groupby(['EDUCATION', 'DEFAULT']).size().unstack(1)

default_education.plot(kind='bar', stacked = True, title = 'DEFAULT BY EDUCATION LEVEL', mark_right = True)

default_education_2 = default_education

default_education_2['PERCENTAGE'] = (default_education[1]/(default_education[0] + default_education[1])) 

default_education_2


# Let's analyze the age distribution and the default rate by age.

default_age = credit['AGE'].dropna()

default_age_dist = sns.distplot(default_age) 
default_age_dist.set_title("Distribution of Default Rate by Ages")


default_age_percentage = credit.groupby(['AGE', 'DEFAULT']).size().unstack(1)

default_age_percentage = default_age_percentage.dropna()

default_age_percentage.plot(kind='bar', stacked = True, title = 'DEFAULT BY AGE', mark_right = True)

default_age_percentage_2 = default_age_percentage

default_age_percentage_2['PERCENTAGE'] = (default_age_percentage[1]/(default_age_percentage[0] + default_age_percentage[1])) 

default_age_percentage_2



# Let's analyze the distribution of age per gender.

fig = sns.FacetGrid(credit, hue='SEX', aspect=4)
fig.map(sns.kdeplot, 'AGE', shade=True)
oldest = credit['AGE'].max()
fig.set(xlim=(0,oldest))
fig.set(title='Distribution of Age Grouped by Gender')
fig.add_legend()


credit.describe()

credit.info()


# Creating a New column.

credit_3 = credit.copy()

# client_quality = [0 for x in range(30000)]

# client_quality = []

# credit_3['client_quality'] = client_quality

credit_3['client_quality'] = credit_3.iloc[:,6:12].sum(axis=1)

credit_3['client_quality_2'] = credit_3.iloc[:,6]*2 + credit_3.iloc[:,7]*1.8 + credit_3.iloc[:,8]*1.6 + credit_3.iloc[:,9]*1.4 + credit_3.iloc[:,10]*1.2 + credit_3.iloc[:,11]*1 

credit_3['client_quality_3'] = np.where((credit_3['client_quality'] < 0) & (credit_3['client_quality_2'] < 0), 'SOLVENT', 'DEFAULTER')

credit_3['client_quality_4'] = np.where(credit_3['client_quality_2'] < 2, 'SOLVENT', 'DEFAULTER')




# Creating Models.-------------------------------------------------------


# Model 1-------------------------

# features

features_model_1 = credit_3.iloc[:,[2,6,7,8,9,10,11,-4,-3]]

features_model_1['SEX'] = features_model_1['SEX'].astype('category')

features_model_1.head()

# dependent variable

depVar_model_1 = credit_3['DEFAULT']


#Training Set (Feature Space: X Training)

X_train_model_1 = (features_model_1[: 30000])
X_train.head()


#Dependent Variable Training Set (y Training)

y_train_model_1 = depVar_model_1[: 30000]
y_train_model_1_count = len(y_train_model_1.index)
print('The number of observations in the Y training set are:',str(y_train_model_1_count))
y_train_model_1.head()


#Testing Set (X Testing)

X_test_model_1 = features_model_1[-6000:]
X_test_model_1_count = len(X_test_model_1.index)
print('The number of observations in the feature testing set is:',str(X_test_model_1_count))
print(X_test.head())


#Ground Truth (y_test) 
y_test_model_1 = depVar_model_1[-6000:]
y_test_model_1_count = len(y_test_model_1.index)
print('The number of observations in the Y training set are:',str(y_test_model_1_count))
y_test.head()


X_train_model_1, X_test_model_1, y_train_model_1, y_test_model_1 = train_test_split(X_train_model_1, y_train_model_1, test_size=0.3, random_state=1816)

X_train_model_1.shape, X_test_model_1.shape


# modelLinear_1 = LinearRegression(n_jobs=10)


# Models.

modelGB_1 = GradientBoostingClassifier()

modelSVM_1 = SGDClassifier()

modelSVM_X = SGDClassifier(loss='hinge',class_weight='balanced')
clf = modelSVM_X.fit(X_train_model_1, y_train_model_1)
calibrator_model1 = CalibratedClassifierCV(clf, cv='prefit')
model1 =calibrator_model1.fit(X_train_model_1, y_train_model_1)

modelRF_1 = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

modelADA_1 = AdaBoostClassifier()

ModelDT_1 = DecisionTreeClassifier()

modelKNN_1 = KNeighborsClassifier()

modelXGBClassifier_1 = XGBClassifier()


# Training the models.

y_train_model_1 = y_train_model_1.astype("category") 

y_test_model_1 = y_test_model_1.astype("category") 


# Gradient Bossting.

modelGB_1.fit(X_train_model_1,y_train_model_1)


# SVM - SGD Classifier.

modelSVM_1.fit(X_train_model_1,y_train_model_1)

model1.fit(X_train_model_1,y_train_model_1)


# Decision Tree.

ModelDT_1.fit(X_train_model_1,y_train_model_1)

# Random Forest.

modelRF_1.fit(X_train_model_1,y_train_model_1)

# ADA Classifier.

modelADA_1.fit(X_train_model_1,y_train_model_1)

# KNN.

modelKNN_1.fit(X_train_model_1,y_train_model_1)

# XGBClassifier

X_train_model_1_bool = X_train_model_1.copy()
X_train_model_1_bool['SEX'] = X_train_model_1_bool['SEX'].astype('bool')

modelXGBClassifier_1.fit(X_train_model_1_bool,y_train_model_1)



modelGB_1.score(X_train_model_1,y_train_model_1)

ModelDT_1.score(X_train_model_1,y_train_model_1)

modelRF_1.score(X_train_model_1,y_train_model_1)

modelSVM_1.score(X_train_model_1,y_train_model_1)

model1.score(X_train_model_1,y_train_model_1)

modelADA_1.score(X_train_model_1,y_train_model_1)
           
modelKNN_1.score(X_train_model_1,y_train_model_1)

modelXGBClassifier_1.score(X_train_model_1_bool,y_train_model_1)



# Making predictions.

predictions_modelGB_1 = modelGB_1.predict(X_test_model_1)

predictions_ModelDT_1 = ModelDT_1.predict(X_test_model_1)

predictions_modelRF_1 = modelRF_1.predict(X_test_model_1)

predictions_modelSVM_1 = modelSVM_1.predict(X_test_model_1)

predictions_model1 = model1.predict(X_test_model_1)

predictions_modelADA_1 = modelADA_1.predict(X_test_model_1)

predictions_modelKNN_1 = modelKNN_1.predict(X_test_model_1)


X_test_model_1_bool = X_test_model_1.copy()
X_test_model_1_bool['SEX'] = X_test_model_1_bool['SEX'].astype('bool')

predictions_modelXGBClassifier_1 = modelXGBClassifier_1.predict(X_test_model_1_bool)




# Evaluating the results.

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test_model_1, predictions_modelGB_1)

confusion_matrix(y_test_model_1, predictions_ModelDT_1)

confusion_matrix(y_test_model_1, predictions_modelRF_1)

confusion_matrix(y_test_model_1, predictions_modelSVM_1)

confusion_matrix(y_test_model_1, predictions_model1)

confusion_matrix(y_test_model_1, predictions_modelADA_1)

confusion_matrix(y_test_model_1, predictions_modelKNN_1)

confusion_matrix(y_test_model_1, predictions_modelXGBClassifier_1)


# ROC Curve.

predictions_model1_probabilities = model1.predict_proba(X_test_model_1)[:,1]

fpr, tpr, _ = roc_curve(y_test_model_1, predictions_model1_probabilities)
roc_auc_model1 = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc_model1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Calculating TP,FP, TN, FN manually.

TP = np.sum(np.logical_and(predictions_model1 == 0, y_test_model_1 == 0))
TN = np.sum(np.logical_and(predictions_model1 == 1, y_test_model_1 == 1))
FP = np.sum(np.logical_and(predictions_model1 == 0, y_test_model_1 == 1))
FN = np.sum(np.logical_and(predictions_model1 == 1, y_test_model_1 == 0))
pred_model1 = len(predictions_model1)
pred_model1

print('True Positives: {}'.format(TP))
print('False Positive: {}'.format(FP))
print('True Negative: {}'.format(TN))
print('False Negative: {}'.format(FN))
print('Precision: {}'.format(round(TP/(TP+FP),2)))
print('Recall: {}'.format(round(TP/(TP+FN),2)))
print('Problematic ratio: {}'.format(round(FN/(FN+TP),2)))


# Model 2-------------------------

credit_4 = credit.copy()

credit_4['client_quality'] = credit_4.iloc[:,6:12].sum(axis=1)

credit_4['client_quality_2'] = credit_4.iloc[:,6]*2 + credit_4.iloc[:,7]*1.8 + credit_4.iloc[:,8]*1.6 + credit_4.iloc[:,9]*1.4 + credit_4.iloc[:,10]*1.2 + credit_4.iloc[:,11]*1 

credit_4['client_quality_3'] = np.where((credit_4['client_quality'] < 0) & (credit_4['client_quality_2'] < 0), 0, 1)

credit_4['client_quality_4'] = np.where(credit_4['client_quality_2'] < 2, 0, 1)




# features

features_model_2 = credit_4.iloc[:,[2,6,7,8,9,10,11,-4,-3,-2,-1]]

features_model_2.head()

features_model_2['client_quality_3'] = features_model_2['client_quality_3'].astype('category')

features_model_2['client_quality_4'] = features_model_2['client_quality_4'].astype('category')

# dependent variable

depVar_model_2 = credit_4['DEFAULT']


#Training Set (Feature Space: X Training)

X_train_model_2 = (features_model_2[: 30000])


#Dependent Variable Training Set (y Training)

y_train_model_2 = depVar_model_2[: 30000]
y_train_model_2_count = len(y_train_model_2.index)
print('The number of observations in the Y training set are:',str(y_train_model_2_count))
y_train_model_2.head()


X_train_model_2, X_test_model_2, y_train_model_2, y_test_model_2 = train_test_split(X_train_model_2, y_train_model_2, test_size=0.3, random_state=1816)

X_train_model_2.shape, X_test_model_2.shape


# modelLinear_1 = LinearRegression(n_jobs=10)


# Models.

modelGB_2 = GradientBoostingClassifier()
modelSVM_2 = SGDClassifier()
modelRF_2 = RandomForestClassifier(n_estimators=100,random_state=0)
modelADA_2 = AdaBoostClassifier()
ModelDT_2 = DecisionTreeClassifier()
modelKNN_2 = KNeighborsClassifier()
modelXGBClassifier_2 = XGBClassifier()



# Training the models.

y_train_model_2 = y_train_model_2.astype("category")

y_test_model_2 = y_test_model_2.astype("category") 



# Gradient Bossting.

modelGB_2.fit(X_train_model_2,y_train_model_2)


# SVM - SGD Classifier.

modelSVM_2.fit(X_train_model_2,y_train_model_2)


# Decision Tree.

ModelDT_2.fit(X_train_model_2,y_train_model_2)

# Random Forest.

modelRF_2.fit(X_train_model_2,y_train_model_2)

# ADA Classifier.

modelADA_2.fit(X_train_model_2,y_train_model_2)


# KNN Classifier.

modelKNN_2.fit(X_train_model_2,y_train_model_2)

# modelXGBClassifier

X_train_model_2_bool = X_train_model_2.copy()
X_train_model_2_bool['SEX'] = X_train_model_2_bool['SEX'].astype('bool')
X_train_model_2_bool['client_quality_3'] = X_train_model_2_bool['client_quality_3'].astype('bool')
X_train_model_2_bool['client_quality_4'] = X_train_model_2_bool['client_quality_4'].astype('bool')

modelXGBClassifier_2.fit(X_train_model_2_bool,y_train_model_2)



modelGB_2.score(X_train_model_2,y_train_model_2)

ModelDT_2.score(X_train_model_2,y_train_model_2)

modelRF_2.score(X_train_model_2,y_train_model_2)

modelSVM_2.score(X_train_model_2,y_train_model_2)

modelADA_2.score(X_train_model_2,y_train_model_2)

modelKNN_2.score(X_train_model_2,y_train_model_2)

modelXGBClassifier_2.score(X_train_model_2_bool,y_train_model_2)
           


# Making predictions.

predictions_modelGB_2 = modelGB_2.predict(X_test_model_2)

predictions_ModelDT_2 = ModelDT_2.predict(X_test_model_2)

predictions_modelRF_2 = modelRF_2.predict(X_test_model_2)

predictions_modelSVM_2= modelSVM_2.predict(X_test_model_2)

predictions_modelADA_2= modelADA_2.predict(X_test_model_2)

predictions_modelKNN_2= modelKNN_2.predict(X_test_model_2)

X_test_model_2_bool = X_test_model_2.copy()
X_test_model_2_bool['SEX'] = X_test_model_2_bool['SEX'].astype('bool')
X_test_model_2_bool['client_quality_3'] = X_test_model_2_bool['client_quality_3'].astype('bool')
X_test_model_2_bool['client_quality_4'] = X_test_model_2_bool['client_quality_4'].astype('bool')

predictions_modelXGBClassifier_2 = modelXGBClassifier_2.predict(X_test_model_2_bool)



# Evaluating the results.

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test_model_2, predictions_modelGB_2)

confusion_matrix(y_test_model_2, predictions_ModelDT_2)

confusion_matrix(y_test_model_2, predictions_modelRF_2)

confusion_matrix(y_test_model_2, predictions_modelSVM_2)

confusion_matrix(y_test_model_2, predictions_modelADA_2)

confusion_matrix(y_test_model_2, predictions_modelKNN_2)

confusion_matrix(y_test_model_2, predictions_modelXGBClassifier_2)



# Model 3-------------------------

# features

features_model_3 = credit_4.iloc[:,[2,3,-4,-3,-2,-1]]

features_model_3['SEX'] = features_model_3['SEX'].astype('category')

features_model_3['EDUCATION'] = features_model_3['EDUCATION'].astype('category')

features_model_3['client_quality_3'] = features_model_3['client_quality_3'].astype('category')

features_model_3['client_quality_4'] = features_model_3['client_quality_4'].astype('category')

features_model_3.head()

# dependent variable

depVar_model_3 = credit_4['DEFAULT']


#Training Set (Feature Space: X Training)

X_train_model_3 = (features_model_3[: 30000])


#Dependent Variable Training Set (y Training)

y_train_model_3 = depVar_model_3[: 30000]
y_train_model_3_count = len(y_train_model_3.index)
print('The number of observations in the Y training set are:',str(y_train_model_3_count))
y_train_model_3.head()


X_train_model_3, X_test_model_3, y_train_model_3, y_test_model_3 = train_test_split(X_train_model_3, y_train_model_3, test_size=0.3, random_state=1816)

X_train_model_3.shape, X_test_model_3.shape


# modelLinear_1 = LinearRegression(n_jobs=10)


# Models.

modelGB_3 = GradientBoostingClassifier()
modelSVM_3 = SGDClassifier()
modelRF_3 = RandomForestClassifier(n_estimators=100,random_state=0)
modelADA_3 = AdaBoostClassifier(n_estimators=50,learning_rate=1)
ModelDT_3 = DecisionTreeClassifier()
modelKNN_3 = KNeighborsClassifier()
modelXGBClassifier_3 = XGBClassifier()


# Training the models.

y_train_model_3 = y_train_model_3.astype("category") 

y_test_model_3 = y_test_model_3.astype("category") 


# Gradient Bossting.

modelGB_3.fit(X_train_model_3,y_train_model_3)


# SVM - SGD Classifier.

modelSVM_3.fit(X_train_model_3,y_train_model_3)


# Decision Tree.

ModelDT_3.fit(X_train_model_3,y_train_model_3)

# Random Forest.

modelRF_3.fit(X_train_model_3,y_train_model_3)


# ADA Classifier.

modelADA_3.fit(X_train_model_3, y_train_model_3)


# KNN Classifier.

modelKNN_3.fit(X_train_model_3, y_train_model_3)

# XGBClassifier.

X_train_model_3_bool = X_train_model_3.copy()
X_train_model_3_bool['SEX'] = X_train_model_3_bool['SEX'].astype('bool')
X_train_model_3_bool['EDUCATION'] = X_train_model_3_bool['EDUCATION'].astype('bool')
X_train_model_3_bool['client_quality_3'] = X_train_model_3_bool['client_quality_3'].astype('bool')
X_train_model_3_bool['client_quality_4'] = X_train_model_3_bool['client_quality_4'].astype('bool')

modelXGBClassifier_3.fit(X_train_model_3_bool,y_train_model_3)





modelGB_3.score(X_train_model_3,y_train_model_3)

ModelDT_3.score(X_train_model_3,y_train_model_3)

modelRF_3.score(X_train_model_3,y_train_model_3)

modelSVM_3.score(X_train_model_3,y_train_model_3)

modelADA_3.score(X_train_model_3,y_train_model_3)

modelKNN_3.score(X_train_model_3,y_train_model_3)

modelXGBClassifier_3.score(X_train_model_3_bool,y_train_model_3)
           


# Making predictions.

predictions_modelGB_3 = modelGB_3.predict(X_test_model_3)

predictions_ModelDT_3 = ModelDT_3.predict(X_test_model_3)

predictions_modelRF_3 = modelRF_3.predict(X_test_model_3)

predictions_modelSVM_3 = modelSVM_3.predict(X_test_model_3)

predictions_modelADA_3 = modelADA_3.predict(X_test_model_3)

predictions_modelKNN_3 = modelKNN_3.predict(X_test_model_3)


X_test_model_3_bool = X_test_model_3.copy()
X_test_model_3_bool['SEX'] = X_test_model_3_bool['SEX'].astype('bool')
X_test_model_3_bool['EDUCATION'] = X_test_model_3_bool['EDUCATION'].astype('bool')
X_test_model_3_bool['client_quality_3'] = X_test_model_3_bool['client_quality_3'].astype('bool')
X_test_model_3_bool['client_quality_4'] = X_test_model_3_bool['client_quality_4'].astype('bool')

prediction_modelXGBClassifier_3 = modelXGBClassifier_3.predict(X_test_model_3_bool)


# Evaluating the results.

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test_model_3, predictions_modelGB_3)

confusion_matrix(y_test_model_3, predictions_ModelDT_3)

confusion_matrix(y_test_model_3, predictions_modelRF_3)

confusion_matrix(y_test_model_3, predictions_modelSVM_3)

confusion_matrix(y_test_model_3, predictions_modelADA_3)

confusion_matrix(y_test_model_3, predictions_modelKNN_3)

confusion_matrix(y_test_model_3, prediction_modelXGBClassifier_3)



# ROC Curve.

predictions_modelRF_3_probabilities = modelRF_3.predict_proba(X_test_model_3)[:,1]

fpr, tpr, _ = roc_curve(y_test_model_3, predictions_modelRF_3_probabilities)
roc_auc_model1 = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc_model1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()



# Model 4 - Solving the class imbalance problem.------------------


# Merging X_train and y_train

train_3 = X_train_model_3.join(y_train_model_3)

train_3_solvent = train_3[train_3.DEFAULT==0]
train_3_defaulter = train_3[train_3.DEFAULT==1]


train_3_defaulter_upsampled = resample(train_3_defaulter, replace=True, n_samples=16330, random_state=587)

# Combine solvent class with upsampled defaulter class

train_3_upsampled = pd.concat([train_3_solvent, train_3_defaulter_upsampled])

# New class counts

train_3_upsampled.DEFAULT.value_counts()


# Models.

modelGB_4 = GradientBoostingClassifier()
modelSVM_4 = SGDClassifier()
modelRF_4 = RandomForestClassifier(n_estimators = 100, class_weight = "balanced")
modelADA_4 = AdaBoostClassifier(n_estimators=50,learning_rate=1)
ModelDT_4 = DecisionTreeClassifier(max_depth = 5, class_weight = "balanced")
modelKNN_4 = KNeighborsClassifier()


# Training the models.

y_train_model_4 = train_3_upsampled['DEFAULT']

y_train_model_4 = y_train_model_4.astype("category") 

y_test_model_4 = y_test_model_3.copy()

y_test_model_4 = y_test_model_4.astype("category") 

X_train_model_4 = train_3_upsampled.iloc[:,[0,1,2,3,4,5]]

X_train_model_4['SEX'] = X_train_model_4['SEX'].astype('category')

X_train_model_4['EDUCATION'] = X_train_model_4['EDUCATION'].astype('category')

X_train_model_4['client_quality_3'] = X_train_model_4['client_quality_3'].astype('category')

X_train_model_4['client_quality_4'] = X_train_model_4['client_quality_4'].astype('category')


# Gradient Bossting.

modelGB_4.fit(X_train_model_4,y_train_model_4)


# SVM - SGD Classifier.

modelSVM_4.fit(X_train_model_4,y_train_model_4)


# Decision Tree.

ModelDT_4.fit(X_train_model_4,y_train_model_4)

# Random Forest.

modelRF_4.fit(X_train_model_4,y_train_model_4)


# ADA Classifier.

modelADA_4.fit(X_train_model_4, y_train_model_4)


# KNN Classifier.

modelKNN_4.fit(X_train_model_4, y_train_model_4)




modelGB_4.score(X_train_model_4,y_train_model_4)

ModelDT_4.score(X_train_model_4,y_train_model_4)

modelRF_4.score(X_train_model_4,y_train_model_4)

modelSVM_4.score(X_train_model_4,y_train_model_4)

modelADA_4.score(X_train_model_4,y_train_model_4)

modelKNN_4.score(X_train_model_4,y_train_model_4)
           


# Making predictions.

predictions_modelGB_4 = modelGB_4.predict(X_test_model_3)

predictions_ModelDT_4 = ModelDT_4.predict(X_test_model_3)

predictions_modelRF_4 = modelRF_4.predict(X_test_model_3)

predictions_modelSVM_4 = modelSVM_4.predict(X_test_model_3)

predictions_modelADA_4 = modelADA_4.predict(X_test_model_3)

predictions_modelKNN_4 = modelKNN_4.predict(X_test_model_3)


# Evaluating the results.

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test_model_3, predictions_modelGB_4)

confusion_matrix(y_test_model_3, predictions_ModelDT_4)

confusion_matrix(y_test_model_3, predictions_modelRF_4)

confusion_matrix(y_test_model_3, predictions_modelSVM_4)

confusion_matrix(y_test_model_3, predictions_modelADA_4)

confusion_matrix(y_test_model_3, predictions_modelKNN_4)


# ROC Curve.

predictions_modelADA_4_probabilities = modelADA_4.predict_proba(X_test_model_3)[:,1]

fpr, tpr, _ = roc_curve(y_test_model_3, predictions_modelADA_4_probabilities)
roc_auc_model1 = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc_model1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()




# Model 5 -------------------------

# features

credit_5 = credit_4.copy()


credit_5['AGE_GROUP'] = 0 #creates a column of 0
credit_5.loc[((credit_5['AGE'] > 20) & (credit_5['AGE'] < 27)) , 'AGE_GROUP'] = 1
credit_5.loc[((credit_5['AGE'] >= 27) & (credit_5['AGE'] < 38)) , 'AGE_GROUP'] = 2
credit_5.loc[((credit_5['AGE'] >= 38) & (credit_5['AGE'] < 45)) , 'AGE_GROUP'] = 3
credit_5.loc[((credit_5['AGE'] >= 45) & (credit_5['AGE'] < 60)) , 'AGE_GROUP'] = 4
credit_5.loc[(credit_5['AGE'] >= 60) , 'AGE_GROUP'] = 5


credit_5['SEX_AGE_GROUP'] = 0
credit_5.loc[((credit_5.SEX == 1) & (credit_5.AGE_GROUP == 1)) , 'SEX_AGE_GROUP'] = 11 # Male Teen
credit_5.loc[((credit_5.SEX == 2) & (credit_5.AGE_GROUP == 1)) , 'SEX_AGE_GROUP'] = 21 # Female Teen
credit_5.loc[((credit_5.SEX == 1) & (credit_5.AGE_GROUP == 2)) , 'SEX_AGE_GROUP'] = 12 # Male Adult
credit_5.loc[((credit_5.SEX == 2) & (credit_5.AGE_GROUP == 2)) , 'SEX_AGE_GROUP'] = 22 # Female Adult
credit_5.loc[((credit_5.SEX == 1) & (credit_5.AGE_GROUP == 3)) , 'SEX_AGE_GROUP'] = 13 # Male Senior
credit_5.loc[((credit_5.SEX == 2) & (credit_5.AGE_GROUP == 3)) , 'SEX_AGE_GROUP'] = 23 # Female Senior
credit_5.loc[((credit_5.SEX == 1) & (credit_5.AGE_GROUP == 4)) , 'SEX_AGE_GROUP'] = 14 # Male Retiree
credit_5.loc[((credit_5.SEX == 2) & (credit_5.AGE_GROUP == 4)) , 'SEX_AGE_GROUP'] = 24 # Female Retiree
credit_5.loc[((credit_5.SEX == 1) & (credit_5.AGE_GROUP == 5)) , 'SEX_AGE_GROUP'] = 15 # Male Elder.
credit_5.loc[((credit_5.SEX == 2) & (credit_5.AGE_GROUP == 5)) , 'SEX_AGE_GROUP'] = 25 # Female Elder.

credit_5['AGE_GROUP'] = credit_5['AGE_GROUP'].astype('category')

credit_5.AGE_GROUP.hist()


credit_5['SEX_AGE_GROUP'] = credit_5['SEX_AGE_GROUP'].astype('category')

credit_5.SEX_AGE_GROUP.hist()

features_model_5 = credit_5.iloc[:,[2,4,-6,-5,-4,-3,-1]]

features_model_5['SEX'] = features_model_5['SEX'].astype('category')

features_model_5['MARRIAGE'] = features_model_5['MARRIAGE'].astype('category')

features_model_5['client_quality_3'] = features_model_5['client_quality_3'].astype('category')

features_model_5['client_quality_4'] = features_model_5['client_quality_4'].astype('category')


features_model_5.head()



# dependent variable

depVar_model_5 = credit_5['DEFAULT']


#Training Set (Feature Space: X Training)

X_train_model_5 = (features_model_5[: 30000])


#Dependent Variable Training Set (y Training)

y_train_model_5 = depVar_model_5[: 30000]



X_train_model_5, X_test_model_5, y_train_model_5, y_test_model_5 = train_test_split(X_train_model_5, y_train_model_5, test_size=0.3, random_state=1816)

X_train_model_5.shape, X_test_model_5.shape


# modelLinear_1 = LinearRegression(n_jobs=10)


# Models.

modelGB_5 = GradientBoostingClassifier()
modelSVM_5 = SGDClassifier()
modelRF_5 = RandomForestClassifier(n_estimators=100,random_state=0)
modelADA_5 = AdaBoostClassifier(n_estimators=50,learning_rate=1)
ModelDT_5 = DecisionTreeClassifier()
modelKNN_5 = KNeighborsClassifier()



# Training the models.

y_train_model_5 = y_train_model_5.astype("category") 

y_test_model_5 = y_test_model_5.astype("category") 


# Gradient Bossting.

modelGB_5.fit(X_train_model_5,y_train_model_5)


# SVM - SGD Classifier.

modelSVM_5.fit(X_train_model_5,y_train_model_5)


# Decision Tree.

ModelDT_5.fit(X_train_model_5,y_train_model_5)

# Random Forest.

modelRF_5.fit(X_train_model_5,y_train_model_5)


# ADA Classifier.

modelADA_5.fit(X_train_model_5, y_train_model_5)


# KNN Classifier.

modelKNN_5.fit(X_train_model_5, y_train_model_5)




modelGB_5.score(X_train_model_5,y_train_model_5)

ModelDT_5.score(X_train_model_5,y_train_model_5)

modelRF_5.score(X_train_model_5,y_train_model_5)

modelSVM_5.score(X_train_model_5,y_train_model_5)

modelADA_5.score(X_train_model_5,y_train_model_5)

modelKNN_5.score(X_train_model_5,y_train_model_5)
           


# Making predictions.

predictions_modelGB_5 = modelGB_5.predict(X_test_model_5)

predictions_ModelDT_5 = ModelDT_5.predict(X_test_model_5)

predictions_modelRF_5 = modelRF_5.predict(X_test_model_5)

predictions_modelSVM_5 = modelSVM_5.predict(X_test_model_5)

predictions_modelADA_5 = modelADA_5.predict(X_test_model_5)

predictions_modelKNN_5 = modelKNN_5.predict(X_test_model_5)


# Evaluating the results.

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test_model_5, predictions_modelGB_5)

confusion_matrix(y_test_model_5, predictions_ModelDT_5)

confusion_matrix(y_test_model_5, predictions_modelRF_5)

confusion_matrix(y_test_model_5, predictions_modelSVM_5)

confusion_matrix(y_test_model_5, predictions_modelADA_5)

confusion_matrix(y_test_model_5, predictions_modelKNN_5)



# Model 6 - Dealing con Class Imbalance Problem.

# Merging X_train and y_train

train_6 = X_train_model_5.join(y_train_model_5)

train_6_solvent = train_6[train_6.DEFAULT==0]
train_6_defaulter = train_6[train_6.DEFAULT==1]


train_6_defaulter_upsampled = resample(train_6_defaulter, replace=True, n_samples=16330, random_state=587)

# Combine solvent class with upsampled defaulter class

train_6_upsampled = pd.concat([train_6_solvent, train_6_defaulter_upsampled])

# New class counts

train_6_upsampled.DEFAULT.value_counts()


# Models.

modelGB_6 = GradientBoostingClassifier()
modelSVM_6 = SGDClassifier()
modelRF_6 = RandomForestClassifier(n_estimators = 100, class_weight = "balanced")
modelADA_6 = AdaBoostClassifier(n_estimators=50,learning_rate=1)
ModelDT_6 = DecisionTreeClassifier(max_depth = 5, class_weight = "balanced")
modelKNN_6 = KNeighborsClassifier()


# Training the models.

y_train_model_6 = train_6_upsampled['DEFAULT']

y_train_model_6 = y_train_model_6.astype("category") 

y_test_model_5 = y_test_model_5.copy()

y_test_model_5 = y_test_model_5.astype("category") 

X_train_model_6 = train_6_upsampled.iloc[:,[0,1,2,3,4,5,6]]


# Gradient Bossting.

modelGB_6.fit(X_train_model_6,y_train_model_6)


# SVM - SGD Classifier.

modelSVM_6.fit(X_train_model_6,y_train_model_6)


# Decision Tree.

ModelDT_6.fit(X_train_model_6,y_train_model_6)

# Random Forest.

modelRF_6.fit(X_train_model_6,y_train_model_6)


# ADA Classifier.

modelADA_6.fit(X_train_model_6, y_train_model_6)


# KNN Classifier.

modelKNN_6.fit(X_train_model_6, y_train_model_6)




modelGB_6.score(X_train_model_6,y_train_model_6)

ModelDT_6.score(X_train_model_6,y_train_model_6)

modelRF_6.score(X_train_model_6,y_train_model_6)

modelSVM_6.score(X_train_model_6,y_train_model_6)

modelADA_6.score(X_train_model_6,y_train_model_6)

modelKNN_6.score(X_train_model_6,y_train_model_6)
           


# Making predictions.

predictions_modelGB_6 = modelGB_6.predict(X_test_model_5)

predictions_ModelDT_6 = ModelDT_6.predict(X_test_model_5)

predictions_modelRF_6 = modelRF_6.predict(X_test_model_5)

predictions_modelSVM_6 = modelSVM_6.predict(X_test_model_5)

predictions_modelADA_6 = modelADA_6.predict(X_test_model_5)

predictions_modelKNN_6 = modelKNN_6.predict(X_test_model_5)


# Evaluating the results.

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test_model_5, predictions_modelGB_6)

confusion_matrix(y_test_model_5, predictions_ModelDT_6)

confusion_matrix(y_test_model_5, predictions_modelRF_6)

confusion_matrix(y_test_model_5, predictions_modelSVM_6)

confusion_matrix(y_test_model_5, predictions_modelADA_6)

confusion_matrix(y_test_model_5, predictions_modelKNN_6)


# ROC Curve.

predictions_modelADA_6_probabilities = modelADA_6.predict_proba(X_test_model_5)[:,1]

fpr, tpr, _ = roc_curve(y_test_model_5, predictions_modelADA_6_probabilities)
roc_auc_model1 = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc_model1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()






# Gianmarco Insights.


#create dictionary with var importance
vars_imp = {}

for i in range(len(modelRF_3.feature_importances_)):
    vars_imp['var'+str(i+1)] = modelRF_5.feature_importances_[i]

#sort vars_imp by decreasing order
    
import operator

sorted_vars_imp = sorted(vars_imp.items(), key=operator.itemgetter(1), reverse=True)

sorted_vars_imp




# Tunning hyperparameters of SVM.

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import  make_scorer, recall_score
import time

start_time = time.time()

#define model
classifier = SVC(kernel ='linear', C=1, gamma=1)

# DEFINE MODEL AND PERFORMANCE MEASURE
### Recall has been chosen since we want to minimize FN
###recall = TP / (TN+FN)
recall_ = make_scorer(recall_score)

# GRID SEARCH FOR n COMBINATIONS OF PARAMETERS
# grid_list = {"C": np.arange(2, 10, 4),
#              "gamma": np.arange(0.1, 1, 0.4),
#              "kernel": ['linear', 'poly']}

# RBF kernel radial basis function kernel
grid_list_2 = {"C": np.arange(1, 20, 1),
               "gamma": np.arange(0.1, 1, 0.01),
               "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
              }

grid_search = GridSearchCV(classifier, param_grid = grid_list_2, n_jobs = -1, cv = 10, scoring = recall_)
grid_search.fit(X_train, y_train)
print("best params:", grid_search.best_params_, "\n\n")
#grid_search.cv_results_

end_time = time.time()

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

timer(start_time, end_time)

grid_search.cv_results_






