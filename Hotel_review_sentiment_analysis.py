#!/usr/bin/env python
# coding: utf-8

# In[64]:


#Data import and check
import pandas as pd
reviewdata = pd.read_csv('train.csv')
reviewdata.head()


# In[65]:


reviewdata.shape


# In[66]:


reviewdata.info()


# In[67]:


reviewdata.describe().transpose()


# In[68]:


reviewdata.isna().sum()


# In[69]:


import seaborn as sns
sns.countplot(x = 'Is_Response', data = reviewdata, palette='Set2')


# In[70]:


#Removing unnecessary columns
reviewdata.drop(columns= ['User_ID', 'Browser_Used', 'Device_Used'], inplace= True)


# In[71]:


#Data cleaning
import re
import string

def text_clean_1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

cleaned1 = lambda x: text_clean_1(x)


# In[72]:


reviewdata['cleaned_description1'] = pd.DataFrame(reviewdata['Description'].apply(cleaned1))
reviewdata.head()


# In[73]:


def text_clean_2(text):
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

cleaned2 = lambda x: text_clean_2(x)


# In[74]:


reviewdata['cleaned_description2'] = pd.DataFrame(reviewdata['cleaned_description1'].apply(cleaned2))
reviewdata.head()


# In[75]:


#model training
from sklearn.model_selection import train_test_split

X = reviewdata.cleaned_description2
y = reviewdata.Is_Response

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 225)

print('X_train :', len(X_train))
print('X_test  :', len(X_test))
print('y_train :', len(y_train))
print('y_test  :', len(y_test))


# In[76]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver = "lbfgs")


# In[77]:


#pipeline
from sklearn.pipeline import Pipeline

model = Pipeline([('vectorizer',tvec),('classifier',clf2)])
model.fit(X_train, y_train)


# In[78]:


#model predictions
from sklearn.metrics import confusion_matrix

predictions = model.predict(X_test)
confusion_matrix(predictions, y_test)


# In[79]:


from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Accuracy : ", accuracy_score(predictions, y_test))
print("Precision : ", precision_score(predictions, y_test, average = 'weighted'))
print("Recall : ", recall_score(predictions, y_test, average = 'weighted'))


# In[87]:


example = ["The team at Novotel Ahmedabad strives to ensure the highest standards of safety and hygiene yet make every instance memorable for you! I am glad that my team has found a place in your excellent comments."]
result = model.predict(example)

print(result)


# In[90]:


# saving model
import pickle

Pkl_Filename = "sentiment_analysis_model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)

