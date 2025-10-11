#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# In[7]:


df = pd.read_csv("wildfires_training.csv")  
df.head(10)


# In[8]:


columns = ['fire','year','temp','humidity','rainfall','drought_code','day','month','wind_speed']


# In[9]:


#df = df[columns]

# Define features and target

X = df[['year','temp','humidity','rainfall','drought_code','buildup_index','day','month','wind_speed']]

y = df['fire']
#y.loc[:, 'fire'] = X['fire'].map({'yes': 1, 'no': 0})
# Train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

# Train baseline Random Forest

rf_classifier = RandomForestClassifier(n_estimators=100, min_samples_leaf=5)


# In[10]:


rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

sample = X_test.iloc[0:1]
prediction = rf_classifier.predict(sample)

sample_dict = sample.iloc[0].to_dict()
print(f"\nSample fire: {sample_dict}")
print(f"Predicted Survival: {'Fire' if prediction[0] == 1 else 'No Fire'}")


# In[ ]:




