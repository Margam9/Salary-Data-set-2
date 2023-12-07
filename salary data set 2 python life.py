#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


df = pd.read_csv("Salary_Data2.csv")


# In[8]:


df.head(5)


# In[9]:


df.tail(4)


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.corr()


# In[13]:


df.nunique()


# In[15]:


df.isnull().sum


# In[22]:


df.hist("YearsExperience",grid= False,color="black")


# In[20]:


df.hist('Salary')


# In[28]:


df.boxplot('Salary',vert=False)


# In[31]:


df.boxplot('YearsExperience',vert=False)


# In[34]:


##Splitting into 2 variables 

X = df[['YearsExperience']]
Y = df[['Salary']]


# In[35]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_


# In[36]:


LR.coef_


# In[44]:


LR.predict(X)
Y_predict = LR.predict(X)
Y_predict


# In[45]:


from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y,Y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(Y,Y_predict) 
print('mean_squared_error :',mse.round(2))
print('root_mean_squared_error :',rmse.round(2))
print('R_squared:',r2)


# In[46]:


import statsmodels.api as sma
X_new = sma.add_constant(X)
LR2 = sma.OLS(Y,X_new).fit()
LR2.summary()


# In[47]:


LR2.rsquared_adj


# In[ ]:




