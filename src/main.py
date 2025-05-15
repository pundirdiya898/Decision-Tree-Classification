# IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate Dataset
from sklearn.datasets import make_classification
# without coefficient of underline model
X,y = make_classification(n_samples=1000,n_features=5,n_clusters_per_class=1,n_classes=2,random_state=2529)

# Get first five rows of target variable (y) and features (X)
X[0:5]
y[0:5]

# Get shape of dataframe
X.shape,y.shape

# Get train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

# Get Decision Tree Classification Model Train
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

# Get Model Prediction
y_pred = model.predict(X_test)
y_pred.shape
y_pred

# Get Model Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))

# Hyperparameter Tunning : Grid Search
from sklearn.model_selection import GridSearchCV
parameters = {'criterion':['gini','entropy'],'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,30,40,50,70,90,120,150]}
gridsearch = GridSearchCV(DecisionTreeClassifier(),parameters)
gridsearch.fit(X_train,y_train)
gridsearch.best_params_
gridsearch.best_score_
gridsearch.best_estimator_
y_pred_grid = gridsearch.predict(X_test)
confusion_matrix(y_test,Y_pred_grid)
print(classification_report(y_test,y_pred_grid))
