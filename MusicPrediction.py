# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 12:32:03 2019

@author: Spikee...Saksham
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.utils import shuffle
from sklearn import metrics
import matplotlib.pyplot as plt


data = pd.read_csv("dataset.csv", header='infer')
data= shuffle(data, random_state=10)

data.describe()

# =============================================================================
# Determing the Features and Labels
# =============================================================================

independent_cols =  ['danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'speechiness', 'tempo', 'time_signature', 'valence']
X=data[independent_cols]

#X = data.iloc[:,0:13].replace(0,np.NaN)
#X=X.fillna(X.median())

#X = data.iloc[:, 0:13]
y = data.like

#Normalizing the Columns
from sklearn import preprocessing

x=X.values
min_scaler =preprocessing.MinMaxScaler()
x_scaled = min_scaler.fit_transform(x)
X=pd.DataFrame(x_scaled)

# =============================================================================
# Splitting the Data into training and test set
# =============================================================================
splitLogistic = int(0.7*len(data))
splitKNN=int(0.85*len(data))


X_train, X_test, y_train, y_test = X[:splitLogistic], X[splitLogistic:], y[:splitLogistic], y[splitLogistic:]
X_trainKNN, X_testKNN, y_trainKNN, y_testKNN = X[:splitKNN], X[splitKNN:], y[:splitKNN], y[splitKNN:]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# =============================================================================
# Models for Prediction
# =============================================================================

#LogisticRegression.... 69, 65, 74
from sklearn.linear_model import LogisticRegression
modelLogistic = LogisticRegression(C=0.9, verbose=2, random_state=10, solver='newton-cg', multi_class='multinomial')


#Tree ... 66.66
#from sklearn import tree
#model=tree.DecisionTreeClassifier()


#KNN....75 76 80  @85%
from sklearn.neighbors import KNeighborsClassifier
modelKNN=KNeighborsClassifier(n_neighbors= 25, algorithm='kd_tree')


#RandomForest...74.59
#from sklearn.ensemble import RandomForestClassifier
#model=RandomForestClassifier(verbose=2,bootstrap=True, random_state=4)


# =============================================================================
# Fitting it into the Model
# =============================================================================
modelLogistic=modelLogistic.fit(X_train,y_train)
modelKNN=modelKNN.fit(X_trainKNN, y_trainKNN)
#
#from sklearn.model_selection import cross_val_score
#scores= cross_val_score(model, X, y, cv=25, scoring='f1_macro' )


# =============================================================================
# Predicting the Labels
# =============================================================================
predictedLogistic = modelLogistic.predict(X_test)
predictedLogistic_y= modelLogistic.predict(X_train)

predictedKNN = modelKNN.predict(X_testKNN)
predictedKNN_y= modelKNN.predict(X_trainKNN)
#print(predicted)


# =============================================================================
# To Calculate Probability
# =============================================================================
probabilityLogistic = modelLogistic.predict_proba(X_test)[::,1]
#print(probabilityLogistic)

probabilityKNN = modelKNN.predict_proba(X_testKNN)[::,1]
#print(probabilityKNN)


# =============================================================================
# Calculating the Confusion Matrix
# =============================================================================
cnf_matrixLogistic = metrics.confusion_matrix(y_test, predictedLogistic)
print("\n",cnf_matrixLogistic)

cnf_matrixKNN = metrics.confusion_matrix(y_testKNN, predictedKNN)
print("\n",cnf_matrixKNN)


# =============================================================================
# Calculating Scores
# =============================================================================

#Results for LogisticRegression
print("Results For Logistic Regression")
scoreLogistic=modelLogistic.score(X_test,y_test)
print("\nScore", scoreLogistic)

print("\nAccuracy Test: ",round(metrics.accuracy_score(y_test, predictedLogistic),2))
print("Accuracy Train: ",round(metrics.accuracy_score(y_train, predictedLogistic_y),2))

print("\nPrecision:",metrics.precision_score(y_test, predictedLogistic))
print("Recall:",metrics.recall_score(y_test, predictedLogistic))

auc = metrics.roc_auc_score(y_test, probabilityLogistic)
print("AUC", round(auc,3))

result=metrics.classification_report(y_test, predictedLogistic)
print("\n", result)

#Results for KNN
print("Results For KNN")

scoreKNN=modelKNN.score(X_testKNN,y_testKNN)
print("\nScore", scoreKNN)

print("\nAccuracy Test: ",round(metrics.accuracy_score(y_testKNN, predictedKNN),2))
print("Accuracy Train: ",round(metrics.accuracy_score(y_trainKNN, predictedKNN_y),2))

print("\nPrecision:",metrics.precision_score(y_testKNN, predictedKNN))
print("Recall:",metrics.recall_score(y_testKNN, predictedKNN))

auc = metrics.roc_auc_score(y_testKNN, probabilityKNN)
print("AUC", round(auc,3))

result=metrics.classification_report(y_testKNN, predictedKNN)
print("\n", result)


# =============================================================================
# Plotting ROC Curve
# =============================================================================

print("ROC Curve for Logistic Regression")
fpr, tpr, _ = metrics.roc_curve(y_test, probabilityLogistic)
plt.plot(fpr, tpr)
plt.show()


#W,b = model.coef_, model.intercept_

#plt.scatter(X_train.iloc[::,1], X_train.iloc[::,9], c=y_train.values)
#ax=plt.gca()
#ax.autoscale= False
#xvals= np.array(ax.get_xlim())
#yvals= -(xvals *W[0][0]+b)/ W[0][1]
#plt.plot(xvals, yvals)
#plt.show()




