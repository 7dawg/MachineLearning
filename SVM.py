#!/usr/bin/env python
# coding: utf-8

# In[71]:


import sklearn
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


# In[72]:


df = pd.read_csv("wildfires_training.csv")  
df.head(10)


# In[73]:


#df = df[columns]

# Define features and target

X = df[['year','temp','humidity','rainfall','drought_code','buildup_index','day','month','wind_speed']]

y = df['fire']
#y.loc[:, 'fire'] = X['fire'].map({'yes': 1, 'no': 0})
# Train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state= 42)

model = svm.SVC(kernel='linear', gamma=1, C=5)
model.fit(X_train, y_train)

# Predict using the SVM model
predictions = model.predict(X)

# Evaluate the predictions
accuracy = model.score(X, y)
print("Accuracy of SVM:", accuracy)


# In[74]:




y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

sample = X_test.iloc[0:1]
prediction = model.predict(sample)

sample_dict = sample.iloc[0].to_dict()
print(f"\nSample fire: {sample_dict}")
print(f"Predicted Survival: {'Fire' if prediction[0] == 1 else 'No Fire'}")


# In[75]:


# Create a kernel support vector machine model
rsvm = svm.SVC(kernel='rbf', gamma=1, C=5)
rsvm.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = rsvm.score(X_test, y_test)
print('Accuracy:', accuracy)


# In[76]:


# Predict using the SVM model
predictions = rsvm.predict(X)

# Evaluate the predictions
accuracy = rsvm.score(X, y)
print("Accuracy of SVM:", accuracy)


# In[ ]:




