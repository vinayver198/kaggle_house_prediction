# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:53:35 2019

@author: vinayver
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.model_selection import learning_curve	
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')


train_dataset = pd.read_csv('train.csv',encoding='utf-8')

crosstab = pd.crosstab(train_dataset['OverallCond'], train_dataset['BldgType'])
stats.chi2_contingency(crosstab)



# Explore the dataset variables
Info = train_dataset.info()
Describe = train_dataset.describe()

# Let's find the quantitative and qualitative features
print('No. of categorical attributes: ',
      train_dataset.select_dtypes(exclude = ['int64','float64']).columns.size)
print('No. of numerical attributes: ',
      train_dataset.select_dtypes(exclude = ['object']).columns.size)


#missing data
total = train_dataset.isnull().sum().sort_values(ascending=False)
percent = (train_dataset.isnull().sum()/train_dataset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# Let's drop the features that are having missing more than 75 %
train_dataset = train_dataset.drop((missing_data[missing_data['Percent']*100 > 75]).index,1)


#descriptive statistics summary
train_dataset['SalePrice'].describe()

# Lets check the normality of our independent variable
sns.distplot(train_dataset['SalePrice'],fit=norm);

#skewness and kurtosis
print("Skewness: %f" % train_dataset['SalePrice'].skew())
print("Kurtosis: %f" % train_dataset['SalePrice'].kurt())

# Observation: Since the sale price is suffering positive skewness
# and leptokurtic so we will transform it.

train_dataset['SalePrice'] = np.log(train_dataset['SalePrice'] + 1)

# let's observe again
#skewness and kurtosis
print("Skewness: %f" % train_dataset['SalePrice'].skew())
print("Kurtosis: %f" % train_dataset['SalePrice'].kurt())


#correlation matrix
corrmat = train_dataset.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


# saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_dataset[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

sns.set_style('whitegrid')
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_dataset[cols])
plt.show()

# Conclusion from heat map
# Variables to be dropped = Id,
# MSSubClass,YearRemodAdd,YrSold,
# EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,miscVal,MoSold,
# BsmtHalfBath,BsmtFullbath,LowQualFinSF,
# BsmttUnfSF,BsmtFinSF2

train_dataset.drop(columns =['MSSubClass','YearRemodAdd',
                             'YrSold','EnclosedPorch','ScreenPorch',
                             'PoolArea','MiscVal',
                             'MoSold','BsmtHalfBath','BsmtFullBath',
                             'LowQualFinSF','BsmtFinSF2',
                             'BsmtUnfSF'
                             ],inplace =True)
#OverallCond,KitchenAbvGr,BedroomAbvGr,

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_dataset[cols], size = 2)
plt.show();


# Lets remove the outliers from dataset
#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([train_dataset['SalePrice'], train_dataset[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#deleting points
train_dataset.sort_values(by = 'GrLivArea', ascending = False)[:2]
train_dataset = train_dataset.drop(train_dataset[train_dataset['Id'] == 1299].index)
train_dataset = train_dataset.drop(train_dataset[train_dataset['Id'] == 524].index)

#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([train_dataset['SalePrice'], train_dataset[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


#histogram and normal probability plot
sns.distplot(train_dataset['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_dataset['SalePrice'], plot=plt)



#histogram and normal probability plot
sns.distplot(train_dataset['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_dataset['GrLivArea'], plot=plt)

#data transformation
train_dataset['GrLivArea'] = np.log(train_dataset['GrLivArea'] + 1)


#transformed histogram and normal probability plot
sns.distplot(train_dataset['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_dataset['GrLivArea'], plot=plt)


#histogram and normal probability plot
sns.distplot(train_dataset['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_dataset['TotalBsmtSF'], plot=plt)

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
train_dataset['HasBsmt'] = pd.Series(len(train_dataset['TotalBsmtSF']), index=train_dataset.index)
train_dataset['HasBsmt'] = 0 
train_dataset.loc[train_dataset['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
train_dataset.loc[train_dataset['HasBsmt']==1,'TotalBsmtSF'] = np.log(train_dataset['TotalBsmtSF'] + 1)
#histogram and normal probability plot
sns.distplot(train_dataset[train_dataset['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_dataset[train_dataset['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)



# Imputing the values
train_dataset['Electrical'] = train_dataset['Electrical'].fillna(train_dataset['Electrical'].mode()[0])
train_dataset['MasVnrArea'] = train_dataset['MasVnrArea'].fillna(int(0))
train_dataset['GarageYrBlt'] = train_dataset['GarageYrBlt'].fillna(train_dataset['GarageYrBlt'].mode()[0])
train_dataset['LotFrontage'] = train_dataset['LotFrontage'].fillna(train_dataset['LotFrontage'].mode()[0])
for iterator in ['FireplaceQu','GarageCond','GarageType','GarageFinish','GarageQual','BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual','MasVnrType']:
    train_dataset[iterator] = train_dataset[iterator].fillna(train_dataset[iterator].mode()[0])




y = train_dataset['SalePrice'].values
train_dataset.drop(columns=['Id','SalePrice'],inplace=True)
train_dataset.drop(columns=['Utilities'],inplace=True)



columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'OverallCond'
        , 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 
        'SaleType', 'SaleCondition', 'Electrical', 'Heating','OverallCond')

#sns.barplot(x='Electrical',y='SalePrice',data=train_dataset) # shows mean or avg for category
#sns.violinplot(x='Electrical',y='SalePrice',data=train_dataset) # shows mean or avg for category
#sns.swarmplot(x='Electrical',y='SalePrice',data=train_dataset) # shows mean or avg for category
#sns.boxplot(x='Electrical',y='SalePrice',data=train_dataset) # shows mean or avg for category
#sns.countplot(x = 'Electrical',data =  train_dataset)

for c in columns:
    train_dataset[c]=train_dataset[c].astype('category')
   

# Let's remove the skewness from numerical features having skewness > 0.5
for iterator in list(train_dataset.columns):
    if iterator not in columns:
        if  train_dataset[iterator].skew() > 0.5:
            print(train_dataset[iterator].skew())
            train_dataset[iterator] =  np.log(train_dataset[iterator] + 1)
            print(train_dataset[iterator].skew())
            print(iterator)
            
       
              

#from sklearn.preprocessing import LabelEncoder
#for c in columns:
#    lbl = LabelEncoder()
#    lbl.fit(list(train_dataset[c].values))
#    train_dataset[c] = lbl.transform(list(train_dataset[c].values))








def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    print(train_sizes)
    print(train_scores)
    print(test_scores)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    return plt





X = pd.get_dummies(train_dataset,drop_first=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_numerical = X_train.select_dtypes(exclude = ['uint8'])
scaled_numerical_features = sc_X.fit_transform(X_train_numerical.values)

labelled_features =  X_train.select_dtypes(exclude = ['int64','float64']).values

transformed_X_train = np.concatenate((scaled_numerical_features, labelled_features), axis=1)


X_test_numerical = X_test.select_dtypes(exclude = ['uint8'])
scaled_numerical_features_test = sc_X.transform(X_test_numerical.values)

labelled_features_test =  X_test.select_dtypes(exclude = ['int64','float64']).values

transformed_X_test = np.concatenate((scaled_numerical_features_test, labelled_features_test), axis=1)



from sklearn.decomposition import PCA
pca = PCA(n_components = 150)
X_train = pca.fit_transform(transformed_X_train)
X_test = pca.transform(transformed_X_test)

explained_variance = pca.explained_variance_ratio_ > 0.001

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)


linear_regressor = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,normalize=False)
linear_regressor.fit(X_train,y_train)
linear_ypred = linear_regressor.predict(X_test)
linear_regressor.score(X_test,y_test)
r2_score(y_test,linear_ypred)
mean_squared_error(y_test,linear_ypred)


# Plot learning curves
title = "Learning Curves (Linear Regression)"
plot_learning_curve(linear_regressor, title, X_train, y_train, ylim=(0.0, 1.01), n_jobs=1)




random_regressor =  RandomForestRegressor()
param_grid = { 
    'n_estimators': [200, 500,1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,10,11,12,13],
    'criterion' :['mse', 'mae']
}
grid = GridSearchCV(random_regressor, param_grid=param_grid, cv=5)
grid.fit(X_train,y_train)
random_regressor =  RandomForestRegressor(criterion='mse',max_depth=8,max_features='auto',n_estimators=500)
random_regressor.fit(X_train,y_train)
random_regressor.score(X_test,y_test)

# Plot learning curves
title = "Learning Curves (DecisionTree Regression)"
plot_learning_curve(random_regressor, title, transformed_X_train, y_train, ylim=(0.0, 1.01), n_jobs=1)


svm_regressor = SVR()
svm_regressor.fit(X_train,y_train)
svm_y_pred = svm_regressor.predict(X_test)
svm_regressor.score(X_test,y_test)
plot_learning_curve(linear_regressor, title, X_train, y_train, ylim=(0.0, 1.01), n_jobs=1)

# Using Xgboost
from sklearn.ensemble import GradientBoostingRegressor
lr = GradientBoostingRegressor()
model = lr.fit(X_train, y_train)
lr_y_pred = model.predict(X_test)
print ("R^2 is: \n", model.score(X_test, y_test))
mean_squared_error(y_test,lr_y_pred)
