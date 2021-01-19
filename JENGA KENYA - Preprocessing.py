#!/usr/bin/env python
# coding: utf-8

# ## Import relevant libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import timedelta
from datetime import datetime
from dateutil.relativedelta import relativedelta


# ### Load the Dataset

# In[2]:


df = pd.read_csv('B:/My Works/Data Science/Jenga Kenya/dummy_data.csv', parse_dates = True)
df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# ### Preprocessing

# In[5]:


#find the null values
df.isnull().sum()


# In[6]:


# fill the missing vallues for numerical terms - mean
df['Loan_Amount'] = df['Loan_Amount'].fillna(df['Loan_Amount'].mean())
df['Loan_Length'] = df['Loan_Length'].fillna(df['Loan_Length'].mode()[0])


# In[7]:


df['Active'].mode()[0]


# In[9]:


# fill the missing values for categorical and ordinal terms - mode
df['Group ID'] = df['Group ID'].fillna(df['Group ID'].mode()[0])
df['Client Type'] = df['Client Type'].fillna(df['Client Type'].mode()[0])
df['Active'] = df['Active'].fillna(df['Active'].mode()[0])
df['Date Of Birth'] = df['Date Of Birth'].fillna(df['Date Of Birth'].mode()[0])
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Marital Status'] = df['Marital Status'].fillna(df['Marital Status'].mode()[0])
df['County'] = df['County'].fillna(df['County'].mode()[0])
df['NOK Relationship'] = df['NOK Relationship'].fillna(df['NOK Relationship'].mode()[0])
df['Main Economic Activity'] = df['Main Economic Activity'].fillna(df['Main Economic Activity'].mode()[0])
df['Client Location'] = df['Client Location'].fillna(df['Client Location'].mode()[0])
df['Secondary Economic Activity'] = df['Secondary Economic Activity'].fillna(df['Secondary Economic Activity'].mode()[0])
df['Occupation'] = df['Occupation'].fillna(df['Occupation'].mode()[0])
df['Loan ID'] = df['Loan ID'].fillna(df['Loan ID'].mode()[0])
df['Loan_status'] = df['Loan_status'].fillna(df['Loan_status'].mode()[0])
df['Loan_Type'] = df['Loan_Type'].fillna(df['Loan_Type'].mode()[0])
df['Initial_Loan_Date'] = df['Initial_Loan_Date'].fillna(df['Initial_Loan_Date'].mode()[0])
df['Interest_Rate'] = df['Interest_Rate'].fillna(df['Interest_Rate'].mode()[0])
df['Number_Payments'] = df['Number_Payments'].fillna(df['Number_Payments'].mode()[0])
df['Time_Between_Payments'] = df['Time_Between_Payments'].fillna(df['Time_Between_Payments'].mode()[0])


# In[10]:


df.isnull().sum()


# ### Exploratory Data Analysis
# 
sns.countplot(df['Client Type'])sns.countplot(df['Active'])sns.countplot(df['Gender'])sns.countplot(df['Marital Status'])sns.countplot(df['Loan_Type'])sns.displot(df['Loan_Length'])sns.displot(df['Loan_Amount'])# apply log transformation to the attribute
df['Loan_Amount'] = np.log(df['Loan_Amount'])sns.displot(df['Loan_Amount'])
sns.displot(df[np.isfinite(df['Loan_Amount'])].values)
# ### Create additional attributes
#  - Age
#  - DOB day
#  - DOB month
#  - DOB year
#  - Disbursal day (Initial Loan date)
#  - Client duration
#  - Number of subsequent Loans

# In[11]:


df['Initial_Loan_Date'] = pd.to_datetime(df['Initial_Loan_Date'])
print(df['Initial_Loan_Date'] .dtypes)


# In[12]:


df['Date Of Birth'] = pd.to_datetime(df['Date Of Birth'])
print (df['Date Of Birth'].dtypes)


# In[13]:


#getting the age of the clients
df['time_diff'] = (df['Initial_Loan_Date'] - df['Date Of Birth'])/365
df['age'] = df['time_diff'].dt.days


# In[14]:


df.columns


# In[15]:


df.drop(['Initial_Loan_Date','Date Of Birth','time_diff'], axis=1, inplace=True)


# In[16]:


df.head()


# ## Label Encoding

# In[17]:


pd.unique(df['Loan_status'])


# In[18]:


from sklearn.preprocessing import LabelEncoder
cols = ['Gender','Loan_status','Client Type','Marital Status','County','NOK Relationship','Main Economic Activity','Client Location','Secondary Economic Activity','Occupation']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])


# In[19]:


df.head()


# In[20]:


df.columns


# #### Re_ordering the columns

# In[24]:


#move the targets to the end of the dataset
re_ordered_cols = ['ClientID', 'Group ID', 'Client Type', 'Active', 'Gender',
       'Marital Status','County', 'NOK Relationship',
       'Main Economic Activity', 'Client Location',
       'Secondary Economic Activity', 'Occupation', 'Loan ID', 'Loan_Type',
       'Loan_Amount', 'Interest_Rate', 'Loan_Length', 'Number_Payments',
       'Time_Between_Payments','age','Loan_status',]


# In[25]:


df = df[re_ordered_cols]


# In[26]:


df


# # Train-Test Split

# In[27]:


targets = df[df.columns[-1]]
unscaled = df[df.columns[1:-1]]


# In[28]:


unscaled


# In[29]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(unscaled,targets, test_size=0.25, random_state=42)


# In[30]:


np.savez('jenga_data_train', inputs=x_train, targets=y_train)
np.savez('jenga_data_test', inputs=x_test, targets=y_test)


# In[ ]:




