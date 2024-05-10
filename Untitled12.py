#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


# In[16]:


df = pd.read_csv("loan.csv")


# In[17]:


df.head()


# In[18]:


df.info()


# In[19]:


df.isnull().sum()


# In[20]:


df['loanAmount_log']= np.log(df['LoanAmount'])
df['loanAmount_log'].hist(bins=20)


# In[21]:


df.isnull().sum()


# In[22]:


df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log']=np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)


# In[23]:


df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)

df.LoanAmount =df.LoanAmount.fillna(df.LoanAmount.mean())
df.loanAmount_log =df.loanAmount_log.fillna(df.loanAmount_log.mean())

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

df.isnull().sum() 


# In[24]:


x = df.iloc[:,np.r_[1:5,9:11,13:15]].values
y = df.iloc[:,12].values

x


# In[25]:


y


# In[26]:


print("per of missing gender is %2f%%" %((df['Gender'].isnull().sum()/df.shape[0])*100))


# In[27]:


print("number of people who take loan as group by gender:")
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data=df,palette= 'Set1')


# In[28]:


print("number of people who take loan as group by gender:")
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data=df,palette= 'Set2')


# In[29]:


print("number of people who take loan as group by gender:")
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data=df,palette= 'Set3')


# In[30]:


print("number of people who take loan as group by marital status:")
print(df['Married'].value_counts())
sns.countplot(x='Married',data=df,palette= 'Set1')


# In[31]:


print("number of people who take loan as group by dependents:")
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents',data=df,palette= 'Set1')


# In[32]:


print("number of people who take loan as group by self employed:")
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed',data=df,palette= 'Set1')


# In[33]:


print("number of people who take loan as group by Loanamount:")
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount',data=df,palette= 'Set1')


# In[34]:


print("number of people who take loan as group by Credit history:")
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History',data=df,palette= 'Set1')


# In[35]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=0)

from sklearn.preprocessing import LabelEncoder
Labelencoder_x =LabelEncoder()


# In[36]:


for i in range(0, 5):
    x_train[:,i]=Labelencoder_x.fit_transform(x_train[:,i])
    x_train[:,7]=Labelencoder_x.fit_transform(x_train[:,7])
    
x_train


# In[37]:


Labelencoder_y = LabelEncoder()
y_train = Labelencoder_y.fit_transform(y_train)

y_train


# In[38]:


for i in range(0, 5):
    x_test[:,i] = Labelencoder_x.fit_transform(x_test[:, i])
    x_test[:,7] = Labelencoder_x.fit_transform(x_test[:, 7])
    
x_test


# In[39]:


labelencoder_y = LabelEncoder()

y_test = Labelencoder_y.fit_transform(y_test)

y_test


# In[40]:


from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
x_train =ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)


# In[41]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(x_train, y_train)


# In[42]:


from sklearn import metrics
y_pred =rf_clf.predict(x_test)

print("acc of random forest clf is", metrics.accuracy_score(y_pred, y_test))

y_pred


# In[43]:


from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)


# In[44]:


y_pred = nb_clf.predict(x_test)
print("acc of gaussianNB is %.", metrics.accuracy_score(y_pred, y_test))


# In[45]:


y_pred


# In[46]:


from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train, y_train)


# In[47]:


y_pred = dt_clf.predict(x_test)
print("acc of DT is", metrics.accuracy_score(y_pred, y_test))


# In[48]:


y_pred


# In[49]:


from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier()
kn_clf.fit(x_train, y_train)


# In[50]:


y_pred = kn_clf.predict(x_test)
print("acc of KN is",metrics.accuracy_score(y_pred,y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




