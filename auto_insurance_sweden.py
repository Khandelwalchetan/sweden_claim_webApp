# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:14:24 2019

@author: Chetan
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


#importing the datset
dataset=pd.read_csv("insurance.csv",header=None,names=["No of Claims","Total Payment"])
dataset.head()



#data preprocessing
X=dataset.iloc[:,0].values
#X has to be 2-D array
X=X.reshape((63,1))

y=dataset.iloc[:,1].values

#splitting into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#prediction
y_pred=regressor.predict(X_test)

print(X.shape)




# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[12]]))


