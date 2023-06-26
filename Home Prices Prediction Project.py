#!/usr/bin/env python
# coding: utf-8

# # Goal of Housing Project
# 
# 1. To predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 
# 
# * Metrics:
# 1. Submissions are evaluated on Mean-Squared-Error,Root-Mean-Squared-Error, Mean Absolute Error, R2 (MSE, RMSE, MAE, R2).
# 2. Submission File Format: The file should contain a header and have the following format:
# ~~~
# Id,SalePrice
# 1461,169000.1
# 1462,187724.1233
# 1463,175221
# ~~~~
# 
# 

# In[1]:


from warnings import filterwarnings
filterwarnings('ignore')


# In[2]:


import os
os.chdir('D:/Datasets')


# ## Step 1: Read CSV file

# In[3]:


# import housing project csv file
import pandas as pd
df = pd.read_csv('Housing_training_set.csv')


# In[4]:


# top 5 records
df.head()


# In[5]:


df.info()


# ## Step 2: Check missing values

# In[6]:


s=df.isna().sum()
s[s>0]


# In[7]:


# Column Names from Housing Datset
df.columns


# ## Step 3: Seperate Dependant and Indepedant Features

# In[8]:


# Drop insignificant column from the dataset and seperate values of dependant and independant features as X and Y
# SalePrice ~ Dependant Features

X = df.drop(labels=['Id','SalePrice'],axis=1)
Y = df[['SalePrice']]


# In[9]:


X.head()


# In[10]:


Y.head()


# In[11]:


# Write a function for Categorical and Continous Values Seperation
def catconsep(df):
    cat = list(df.columns[df.dtypes=='object'])
    con = list(df.columns[df.dtypes!='object'])
    return cat,con


# In[12]:


cat,con = catconsep(X)


# In[13]:


cat


# In[14]:


con


# ### Step 4: Build First Pipline for Feature Selection
# 
# * Pipeline :
# 
# 1. Con : Imputer, StandardScaler
# 2. Cat : Imputer, OrdinalEncoder

# In[15]:


# Import sklearn Libraries
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


# In[16]:


# Build Numeric Pipeline
num_pipe = Pipeline(steps=[('Imputer',SimpleImputer(strategy='mean')),
                           ('Scaler',StandardScaler())])
# Build Categorical Pipeline
cat_pipe = Pipeline(steps=[('Imputer',SimpleImputer(strategy='most_frequent')),
                           ('ORD', OrdinalEncoder())])
# Combine Pipeline
pre = ColumnTransformer([('num',num_pipe,con),
                         ('cat',cat_pipe,cat)]) 


# In[17]:


X_pre = pre.fit_transform(X)
X_pre


# In[18]:


cols = pre.get_feature_names_out()
cols


# In[19]:


# combine dataframe
X_pre =pd.DataFrame(X_pre,columns=cols)
X_pre


# In[20]:


X_pre.head()


# In[21]:


## For Feature Selection import Sequentialfeature Selector
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector


# In[22]:


model = LinearRegression()                      # first model of linear regression
# selecting model for forward feature selection
sel = SequentialFeatureSelector(model,direction='forward',n_jobs=-1)                 
sel_features = sel.fit_transform(X_pre,Y)


# In[23]:


sel_cols = sel.get_feature_names_out()
sel_cols


# In[24]:


cols_Len = len(sel_cols)
cols_Len


# In[25]:


## Important Columns
imp_cols =[]
for i in sel_cols:
    s=i.split('__')[1]
    imp_cols.append(s)


# In[26]:


imp_cols                                # Important Columns Seperation


# In[27]:


X_sel = X[imp_cols]
X_sel.head()


# In[28]:


X_sel.shape


# In[29]:


## cat-con seperation for selected features
cat_sel,con_sel = catconsep(X_sel)


# In[30]:


cat_sel


# In[31]:


con_sel


# ## Step 5: Build second Pipeline for training the model
# 1. Con : imputer, StandardScaler
# 2. Cat : imputer, OneHotEncoder

# In[32]:


# numerical pipeline 1
num_pipe1 = Pipeline(steps=([('Imputer',SimpleImputer(strategy='mean')),
                             ('Scaler',StandardScaler())]))
# Categorical Pipeline 2
cat_pipe1 = Pipeline(steps=([('Imputer',SimpleImputer(strategy='constant',fill_value='NotAvailable')),
                             ('OHE',OneHotEncoder(handle_unknown='ignore'))]))
# Combine with Column-Transformer
pre1 = ColumnTransformer([('num',num_pipe1,con_sel),
                          ('cat',cat_pipe1,cat_sel)])


# In[33]:


X_pre1 = pre1.fit_transform(X_sel).toarray()
X_pre1


# In[34]:


cols1 = pre1.get_feature_names_out()
cols1


# In[35]:


# compose Dataframe
X_pre1 = pd.DataFrame(X_pre1, columns=cols1)
X_pre1


# In[36]:


X_pre1.shape


# ## Step 6: Train-Test Split

# In[37]:


# importing train-test-split from sklearn
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X_pre1,Y,test_size=0.2,random_state=21)


# In[38]:


xtrain.shape                    #testing data size


# In[39]:


xtest.shape


# In[40]:


## Build second Linear Regression Model
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(xtrain,ytrain)


# In[41]:


# r2 score evaluation in training
r2_score_tr = model1.score(xtrain,ytrain)
r2_score_tr


# In[42]:


# r2 score evaluation in testing
r2_score_ts = model1.score(xtest,ytest)
r2_score_ts


# In[43]:


## model evaluation Function: 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
def r2_adj_score(xtrain,ytrain,model):
    r = model.score(xtrain,ytrain)
    N = xtrain.shape[0]
    P = xtrain.shape[1]
    num = (1-r)*(1-N)
    den =(N-P-1)
    adj_r2 = 1 - (num/den)
    return adj_r2


def evaluate_model(xtrain,ytrain,xtest,ytest,model):
    ypred_tr = model.predict(xtrain)
    ypred_ts = model.predict(xtest)
    
    # Evaluate training data
    tr_mse = mean_squared_error(ytrain,ypred_tr)
    tr_rmse = tr_mse**(1/2)
    tr_mae = mean_absolute_error(ytrain,ypred_tr)
    tr_r2 = r2_score(ytrain,ypred_tr)
    
    # Evaluate testing data
    ts_mse = mean_squared_error(ytest,ypred_ts)
    ts_rmse = ts_mse**(1/2)
    ts_mae = mean_absolute_error(ytest,ypred_ts)
    ts_r2 = r2_score(ytest,ypred_ts)
    
    # Print the results
    print('Training Results :')
    print(f'MSE : {tr_mse:.2f}')
    print(f'RMSE: {tr_rmse:.2f}')
    print(f'MAE : {tr_mae:.2f}')
    print(f'R2  : {tr_r2:.4f}')
    
    print('\n============================\n')
    
    print('Testing Results :')
    print(f'MSE : {ts_mse:.2f}')
    print(f'RMSE: {ts_rmse:.2f}')
    print(f'MAE : {ts_mae:.2f}')
    print(f'R2  : {ts_r2:.4f}')


# In[44]:


evaluate_model(xtrain,ytrain,xtest,ytest,model1)


# In[45]:


r2_adj_score(xtrain,ytrain,model1)                        #R2 Adj


# ## Step 7 : Ridge and Lasso Tuning
# 1. Alpha value : Hyperparameter
# 2. GridSearchCV

# In[46]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import numpy as np


# In[47]:


scores = cross_val_score(model1,xtrain,ytrain,cv=5,scoring='neg_mean_squared_error')
scores


# In[48]:


r2_scores = cross_val_score(model1,xtrain,ytrain,cv=5,scoring='r2')
r2_scores


# In[49]:


r2_scores.mean()


# In[50]:


mean_scores =scores.mean()
mean_scores


# In[51]:


## calculate Alpha Values
alphas = np.arange(0.1,100,0.1)
# parmeters Dictionary
params ={'alpha':alphas}
params


# In[52]:


## Ridge Model
model2 = Ridge()
gscv1 = GridSearchCV(model2,param_grid=params,cv=5,scoring='neg_root_mean_squared_error')
gscv1.fit(xtrain,ytrain)


# In[53]:


gscv1.best_params_


# In[54]:


gscv1.best_score_


# In[55]:


best_ridge =gscv1.best_estimator_
best_ridge


# In[56]:


evaluate_model(xtrain,ytrain,xtest,ytest,best_ridge)


# In[57]:


r2_Adj_score = r2_adj_score(xtrain,ytrain,best_ridge)
print(f'R2_Adj: {r2_Adj_score:.4f}')


# In[58]:


## Tune Lasso Model
model3 = Lasso()
gscv2 = GridSearchCV(model3, param_grid=params,cv=5,scoring='neg_root_mean_squared_error')
gscv2.fit(xtrain,ytrain)


# In[59]:


gscv2.best_params_


# In[60]:


gscv2.best_score_


# In[61]:


best_lasso = gscv2.best_estimator_
best_lasso


# In[62]:


evaluate_model(xtrain,ytrain,xtest,ytest,best_lasso)


# In[63]:


r2_adj_score(xtrain,ytrain,best_lasso)                #Adjusted R2 Score


# ## Step 8: Read Testing file for House Price Prediction

# In[64]:


df1 = pd.read_csv('Housing.csv')
df1.head()


# In[65]:


df1.shape


# In[66]:


s= df1.isna().sum()
s[s>0]


# In[67]:


xsamp = pre1.transform(df1).toarray()
xsamp


# In[68]:


cols2 = pre1.get_feature_names_out()
cols2


# In[69]:


xsamp = pd.DataFrame(xsamp,columns=cols2)
xsamp


# In[70]:


xsamp.shape


# ## Step 9: Selecting Best Ridge Model with accuracy of 83% for Predicting Housing Prices

# In[71]:


preds=best_ridge.predict(xsamp)
preds


# In[74]:


df2 = df1[['Id']]
df2


# In[75]:


df2['Pred_Sales_Price'] =preds
df2


# In[76]:


df2.to_csv('Predicted_Housing_Prices.csv',index=False)


# In[ ]:




