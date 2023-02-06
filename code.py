#!/usr/bin/env python
# coding: utf-8

# In[11]:


#numpy pandas matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X=[4,5,10,4,10,8,3,2,7,3]
Y=[23,27,22,30,21,23,27,29,30,25]
data=list(zip(X,Y))
data
from sklearn.cluster import KMeans
model=KMeans(n_clusters=2)
model.fit(data)
plt.scatter(X,Y,c=model.labels_)
plt.show()


# In[14]:


import pandas as pd
df1=pd.read_csv("C:/Users/DITU/Downloads/2/3/titanic_dataset.csv")
df1


# In[19]:


df1=df1.dropna()
X=df1["Fare"]
Y=df1["Age"]

data=list(zip(X,Y))
data
from sklearn.cluster import KMeans
model=KMeans(n_clusters=5)
model.fit(data)
plt.scatter(X,Y,c=model.labels_)
plt.show()

