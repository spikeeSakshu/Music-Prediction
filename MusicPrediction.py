# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 12:32:03 2019

@author: Spikee
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.utils import shuffle
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("dataset.csv", header='infer')

data= shuffle(data, random_state=0)

data.corr()


#independent_cols =  ['danceability', 'duration_ms','instrumentalness','speechiness','valence']
#X=data[independent_cols]

X = data.iloc[:,0:13]
y = data.like

#model=LogisticRegression()
model = LogisticRegression(solver='newton-cg', multi_class='multinomial')

split = int(0.7*len(data))

X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]


model=model.fit(X_train,y_train)

#print(pd.DataFrame(zip(X.col_names, np.transpose(model.coef_))))

probability = model.predict_proba(X_test)
print(probability)

predicted = model.predict(X_test)
print(predicted)

cnf_matrix = metrics.confusion_matrix(y_test, predicted)
print(cnf_matrix)

print("Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Precision:",metrics.precision_score(y_test, predicted))
print("Recall:",metrics.recall_score(y_test, predicted))
result=metrics.classification_report(y_test, predicted)
print(result)
d=model.score(X_test,y_test)
print(d)