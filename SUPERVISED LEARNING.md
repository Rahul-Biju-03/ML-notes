## Regression

TYPES:

linear
logistic
decision tree

attributes/features
dependant variables
independant variables

Linear regression
pattern recognition
model

input dataset along with algorithm(such as linear regression) gives different model everytime due to difference in domain used (such as health system,face recognition).
Model is formed as a result of training.
Model studies the pattern of data.
Therefore when new data comes, it will recognise the data.
generalisation-performs well on unseen data.

parameters

## STEPS FOR CODE IN GENERAL

load library
load dataset
EDA
split dataset
create model

### LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
seaborn
pickle

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

from prettytable import PrettyTable

category_encoders- convert labels into numerical
---
## LINEAR REGRESSION

### TYPES OF LINEAR REGRESSION:

single linear regression
y=mx+c
m and c are paramters.

multiple linear regression
y=c+m1x1+m2x2+....

non linear regression
y=g(x/theta)
g()-model , theta-paramters

### COST FUNCTION()

measures performance of model.
finds error
goal_minimise cost fn
we need to minimise the following fn
j=1/n sigma(predi-yi)^2    (sigma=>i=1 to n)

### REGULARISATION OF LINEAR MODELS

1)ridge (l2 regularisation)
2)lasso (l1 regularisation)
3)elastic net (combines l1 and l2)

overfit-w/o regularisation
good fit- with regularisation
regualarisation helps in getting a good fit model.

ridge expresson = loss fn + regularised term
lasso expresson = loss fn + regularised term
create fit predict evaluate model

try other attributes for car.

## LOGISTIC REGRESSION

non linear
while plotting a graph,we have to bent the line at a particular point.We need to find this point.

### EVALUATION METRICS

check n/b

### TICK MARKS
small indicators along axes of graph along with their labels.

### ILOC
integer location

## DECISION TREE

node-feature/attribute
branch-link

### STEPS FOR DECISION TREE

1)create a root node (calculate entropy ,avg info ,info gain)
  check nb for eqns.
  entropy=0 when either all are positive or all are negative.
  value=0 for entropy is good.
  Higher gain value is good. Attribute with highest gain value is selected as the root node.
  
2)Left node
  Recalculate using eqns for all other attributes again w/o root node.
  Select the next attrbute with the highest gain value and put it as the left node.
  
3)Recalculate once again. Attribute with highest gain value is placed at right node.

4)Final decision tree is completed.

### STEPS FOR GINI

1)calculate gini value for all attributes
2)Attribute with lowest gini value is the root node.
3)Repeat for left node and then right node.

### ISSUES IN DECISION TREE

accuracy first increase then decrease.
overfitting

### CODE FOR DECISION TREE

Set criterion=entropy if we want gain value.
or set criterion=gini.

## SVM

find optimal hyperplane (plane that has max. distance from both the classes used)

### TYPES OF SVM

linear- Data set:linearly sperable.
        divides categories using a simple straight lines.
        Is able to divide data into 2 categories using a straight line.
non linear - Data set:not linearly seperable.
             transforms data into 2d.
           
## ENSEMBLE LEARNING

Combines decisions from multiple models to improve performance.
enhances accuracy
mitigates error in individual models

### TYPES OF ENSEMBLE LEARNING

max voting-highest number of votes
averaging - take avg of predictions.
weighted averaging -models are assigned weights. Higher weight-more importance.
soft voting takes the average value(average voting)
hard voting is majority voting

Bagging

## UNSUPERVISED LEARNING

Dont need data to be labeled.
Most of the deep learning is unsupervised.
Egs- 'K means clustering' and 'PCA'
Clustering- group similar data based on certain criteria.

### K MEANS CLUSTERING

## application-
customer segmentation
medicine
psychology

## definition
Data organised into distint groups having centroids(mean values)
k = no of clusters

## steps
1)choose value of k
2)initialise centroids/mean randomly in each of the clusters.
3)calculate euclidean distance of each data pt with all centroids.
4)data pts are assigned to clusters with are nearest to its respective centroids.
5)recalculate mean of each cluster and we will get a new centroid for each cluster. Recalculate euclidean distance again. Due to this data pts may change their clusters.
6)Repeat step 5.

## Mathematical aspect
1)if k=2 => k1 and k2.
2)C1=16,C2=22
3)data pt=15 
  ED= sqrt((15-16)^2)=1 (C1)
  ED=sqrt((15-22)^2)=7  (C2)
4)here data pt 15 is assigned to C1
5)new centroid of C1=(15+15+15)/3=15.33
                  C2=36.25
  ED=sqrt((19-15.33^2)=3.07 (C1)
6)data pts gets changed again and we get new centroids. Calculste ED again.

## How to choose optimal k

improper selection-erroneous assignment

SSW-sum of squared within clusters (inside it)
SSB-sum of squared between clusters
SSW should be least
SSB should be max.

elbow method used-
SSW decrease as no. of clusters decrease. 
do elbow method until u get elbow pt.

