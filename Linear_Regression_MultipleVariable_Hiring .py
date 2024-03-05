#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt 
import numpy as np




# In[2]:


pip install word2number #convert word to number


# In[3]:


from word2number import w2n


# In[4]:


df=pd.read_csv('D:\\UWE\\MachineLearning\\DataSets\\hiring.csv')
df


# In[5]:


df['experience'][0] ='zero'
   


# In[6]:


df['experience'][1] ='one' 


# In[7]:


df['experience']


# In[15]:


num=[0,1,2,3,4,5,6,7]


# In[22]:


for x,i in zip(num,df['experience']):
    print(x,i)
    #w2n.word_to_num(i)
    #print(w2n.word_to_num(i))
    #df['experience'][x]=w2n.word_to_num(i)
    
df['experience']
    


# In[23]:


df


# In[30]:


mean_1=df['test_score(out of 10)'].mean()
print(mean_1)


# In[33]:


import math 
mean_round=math.floor(mean_1) # round the mean
df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(mean_round) # fill mean value to null
df


# In[37]:


lm_1=linear_model.LinearRegression()
lm_1.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])
lm_1.predict([[2,4.0,5]])


# In[ ]:




