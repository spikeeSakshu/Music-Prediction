# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 12:32:03 2019

@author: Spikee
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing

from sklearn.utils import shuffle
from sklearn import metrics
import matplotlib.pyplot as plt


data = pd.read_csv("dataset.csv", header='infer')
data= shuffle(data, random_state=10)


data.corr()

data.describe()


# =============================================================================
# Determing the Features and Labels
# =============================================================================

#independent_cols =  ['danceability', 'duration_ms','instrumentalness','speechiness','valence']
#X=data[independent_cols]

#X = data.iloc[:,0:13].replace(0,np.NaN)
#X=X.fillna(X.median())

X = data.iloc[:,0:13]
y = data.like

#Normalizing the Columns
x=X.values
min_scaler =preprocessing.MinMaxScaler()
x_scaled = min_scaler.fit_transform(x)
X=pd.DataFrame(x_scaled)

# =============================================================================
# Splitting the Data into training and test set
# =============================================================================
split = int(0.7*len(data))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]


# =============================================================================
# Different Models for Prediction
# =============================================================================

#LogisticRegression.... 64.93
model = LogisticRegression(C=1, verbose=2, random_state=10, solver='newton-cg', multi_class='multinomial')

#model=LinearRegression

#Tree ... 66.66
#model=tree.DecisionTreeClassifier()

#KNN....57.24
#model=KNeighborsClassifier(algorithm='brute', )

#RandomForest...74.59
#model=RandomForestClassifier(verbose=2,bootstrap=True, random_state=4)



# =============================================================================
# Fitting it into the Model
# =============================================================================
model=model.fit(X_train,y_train)


# =============================================================================
# Predicting the Labels
# =============================================================================
predicted = model.predict(X_test)
predicted_y= model.predict(X_train)
print(predicted)

#probability = model.predict_proba(X_test)
#print(probability)


# =============================================================================
# Calculating the Confusion Matrix
# =============================================================================
cnf_matrix = metrics.confusion_matrix(y_test, predicted)
#cnf_matrix=np.transpose(cnf_matrix)
print(cnf_matrix)

# =============================================================================
# Calculating other Scores
# =============================================================================
print("Accuracy: test",round(metrics.accuracy_score(y_test, predicted),2))
print("Accuracy Train: test",metrics.accuracy_score(y_train, predicted_y))
print("Precision:",metrics.precision_score(y_test, predicted))
print("Recall:",metrics.recall_score(y_test, predicted))
result=metrics.classification_report(y_test, predicted)
print(result)

d=model.score(X_test,y_test)
print(d)