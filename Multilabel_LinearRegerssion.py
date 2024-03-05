#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('D:\\UWE\\MachineLearning\\DataSets\\ageIncome.csv')
df 


# In[3]:


mean_experiance=df['experience'].mean()
mean_experiance


# In[4]:


import math 
mean_experiance=math.floor(mean_experiance)
mean_experiance


# In[5]:


df['experience']=df['experience'].fillna(mean_experiance)
df['experience']


# In[6]:


df


# In[7]:


lm=linear_model.LinearRegression()
lm.fit(df[['age','experience']],df['income'])


# In[8]:


lm.coef_


# In[9]:


lm.intercept_


# In[10]:


lm.predict([[30,10]])


# In[11]:


lm.predict([[0,0]])


# In[ ]:





# In[ ]:




