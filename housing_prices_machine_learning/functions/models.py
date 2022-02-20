#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

# Load libraries
from pandas import set_option
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

#********************************************************************************************************************

def GetBasedModel():
    '''a function which will instantiate as many models as you wish'''
    
    basedModels = []
    basedModels.append(('LR'   , LogisticRegression()))
    basedModels.append(('KNN'  , KNeighborsClassifier()))
    basedModels.append(('CART' , DecisionTreeClassifier()))
    basedModels.append(('SVM'  , SVC(probability=True)))
    basedModels.append(('RF'   , RandomForestClassifier()))
    #basedModels.append(('ET'   , ExtraTreesClassifier())) 
    #basedModels.append(('LDA'  , LinearDiscriminantAnalysis()))
    #basedModels.append(('NB'   , GaussianNB()))
    #basedModels.append(('AB'   , AdaBoostClassifier()))
    #basedModels.append(('GBM'  , GradientBoostingClassifier()))
    return basedModels

#********************************************************************************************************************

def BasedModels(X_train, y_train,models):
    """
    BasedModels will return the evaluation metric 'accuracy' after performing
    a CV for each of the models
    input:
    X_train
    y_train
    models = array containing the different instantiated models
    
    output:
    names = names of the diff models tested
    results = results of the diff models
    """
    # Test options and evaluation metric
    num_folds = 10
    
    #num_folds =  k_folds
    scoring = 'accuracy'

    results = []
    names = []
    
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds)
        cv_results = cross_val_score(model, X_train,
                                     y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: Accuracy = %f (std = %f)" % (name, 
                                                cv_results.mean(), 
                                                cv_results.std())
        print(msg)
        
    return names, results

#********************************************************************************************************************

def ScoreDataFrame(names,results):
    """
    A function to append the results of the different models
    assessed and put them together in a DF

    """
    def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" 
    
        return float(prc.format(f_val))

    scores = []
    for r in results:
        scores.append(floatingDecimals(r.mean(),4))

    scoreDataFrame = pd.DataFrame({'Model':names, 'Score': scores})
    return scoreDataFrame

#********************************************************************************************************************

def MetricsClas(models,X_train, y_train, X_test, y_test):
    for name, model in models:
        print('-*-'*25)
        print('Assessment of ', str(name), '\n')
        model_fit = model.fit(X_train, y_train)
        Allmetrics(model_fit, X_train, y_train, X_test, y_test)

#********************************************************************************************************************

def GetScaledModel(nameOfScaler):
    """
    Function to define whether we want to apply any preprocessing method to the raw data.
    input:
    nameOfScale  = 'standard' (standardize) or 'minmax'
    """
    
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
        
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()
        
    elif nameOfScaler == 'robustscaler':
        scaler = RobustScaler()

    pipelines = []
    pipelines.append((nameOfScaler+'LR'  , 
                      Pipeline([('Scaler', scaler)
                                ,('LR'  , LogisticRegression())])))
    
    pipelines.append((nameOfScaler+'KNN' , 
                      Pipeline([('Scaler', scaler),('KNN' , 
                                                   KNeighborsClassifier())])))
    pipelines.append((nameOfScaler+'CART', 
                      Pipeline([('Scaler', scaler),('CART', DecisionTreeClassifier())])))
    pipelines.append((nameOfScaler+'SVM' ,
                      Pipeline([('Scaler', scaler),('SVM' , SVC())])))
    pipelines.append((nameOfScaler+'RF'  , 
                      Pipeline([('Scaler', scaler),('RF'  , RandomForestClassifier())])))
    
    #pipelines.append((nameOfScaler+'ET'  , Pipeline([('Scaler', scaler),('ET'  , ExtraTreesClassifier())])  ))
    #pipelines.append((nameOfScaler+'LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])))
    #pipelines.append((nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])))
    #pipelines.append((nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])  ))
    #pipelines.append((nameOfScaler+'GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier())])  ))

    return pipelines 
#********************************************************************************************************************
