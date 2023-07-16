#hyperparameter tunning 

import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import metrics 
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline

from functools import partial


df = pd.read_csv("train.csv")

X = df.drop("price_range" , axis = 1).values
y = df.price_range.values

classifer = ensemble.RandomForestClassifier(n_jobs =-1)
param_grid = {
    "n_estimators": [100,200,300,400] ,
    "max_depth": [1,3,5,7],
    "criterion":["gini" , "entropy"],
}

model = model_selection.GridSearchCV(
    estimator =classifer ,
    param_grid= param_grid,
    scoring = "accuracy",
    verbose= 10 ,
    n_jobs= 1,
    cv =5,
)
model.fit(X,y)
print(model.best_score_)
print(model.best_estimator_.get_params())