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
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


train_dataset = pd.read_csv('train.csv',encoding='utf-8')

# Explore the dataset variables
train_dataset.info()


#descriptive statistics summary
train_dataset['SalePrice'].describe()

#histogram
sns.distplot(train_dataset['SalePrice']);

#skewness and kurtosis
print("Skewness: %f" % train_dataset['SalePrice'].skew())
print("Kurtosis: %f" % train_dataset['SalePrice'].kurt())
#correlation matrix
corrmat = train_dataset.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_dataset[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_dataset[cols], size = 2.5)
plt.show();


#missing data
total = train_dataset.isnull().sum().sort_values(ascending=False)
percent = (train_dataset.isnull().sum()/train_dataset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


#dealing with missing data
train_dataset = train_dataset.drop((missing_data[missing_data['Total'] > 1]).index,1)
train_dataset = train_dataset.drop(train_dataset.loc[train_dataset['Electrical'].isnull()].index)
train_dataset.isnull().sum().max()



#standardizing data
saleprice_scaled = StandardScaler().fit_transform(train_dataset['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

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

#applying log transformation
train_dataset['SalePrice'] = np.log(train_dataset['SalePrice'])

#transformed histogram and normal probability plot
sns.distplot(train_dataset['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_dataset['SalePrice'], plot=plt)


#histogram and normal probability plot
sns.distplot(train_dataset['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_dataset['GrLivArea'], plot=plt)

#data transformation
train_dataset['GrLivArea'] = np.log(train_dataset['GrLivArea'])


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
train_dataset.loc[train_dataset['HasBsmt']==1,'TotalBsmtSF'] = np.log(train_dataset['TotalBsmtSF'])
#histogram and normal probability plot
sns.distplot(train_dataset[train_dataset['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_dataset[train_dataset['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

#scatter plot
plt.scatter(train_dataset['GrLivArea'], train_dataset['SalePrice']);

#scatter plot
plt.scatter(train_dataset[train_dataset['TotalBsmtSF']>0]['TotalBsmtSF'], train_dataset[train_dataset['TotalBsmtSF']>0]['SalePrice']);



y = train_dataset['SalePrice'].values
train_dataset.drop(columns = ['Id','SalePrice'],inplace = True)

train_dataset = pd.get_dummies(train_dataset)

X = train_dataset.iloc[:,1:220].values



#Scaling the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)



from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=4)
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
print(np.sqrt(mse))
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)

import numpy as np
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(regressor, X, y, n_jobs=-1, cv=2, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

import numpy  as np
import pandas as pd

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


# Plot learning curves
title = "Learning Curves (DecisionTree Regression)"
cv = 10
plot_learning_curve(regressor, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1)