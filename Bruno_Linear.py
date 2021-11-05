#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:41:57 2021

@author: brunomorgado
"""

#Import the necessary libraries and methods
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve


#Load dataset
filename = 'titanic.csv'
path = '/Users/brunomorgado/Dropbox/Education/Mac_Education/Centennial_College/Third_semester/AI/Assignments/Assignment4'
fullpath = os.path.join(path,filename)

titanic_bruno = pd.read_csv(fullpath)


#Data exploration
titanic_bruno.head(3)
titanic_bruno.shape
titanic_bruno.dtypes
titanic_bruno.info()

#Display Unique values
titanic_bruno['Sex'].value_counts()

titanic_bruno['Pclass'].value_counts()

#Use a heatmap to visualize missing data
sns.heatmap(titanic_bruno.isna(),yticklabels=False,cbar=False,cmap='viridis')


#Plot bar chart of Survived vs Passenger Class
pclass_survived = pd.crosstab(titanic_bruno['Survived'],titanic_bruno['Pclass'])
pclass_survived.plot(kind='bar')
plt.title('Class_Survivors_Bruno')
plt.ylabel('Number of deceased & survivors per sex')

#Plot bar chart of Survived vs Sex
sex_survived = pd.crosstab(titanic_bruno['Survived'], titanic_bruno['Sex'])
sex_survived.plot(kind='bar')
plt.title('Gender_Survivors_Bruno')
plt.ylabel('Number of deceased and survivors per gender')

#Investigate the number of unique variables in 'Survived'
titanic_bruno.Survived.value_counts()

#Using a scatter matrix to analyze the relationships between variables
pd.plotting.scatter_matrix(titanic_bruno[['Survived', 'Sex', 'Fare', 'Pclass', 'SibSp', 'Parch']], figsize=(17,15), hist_kwds={'bins':10}, alpha=0.1)

#Using Seaborn Pairplot in an attempt to have a better view of the relationship between variables
sns.pairplot(titanic_bruno[['Survived', 'Sex', 'Pclass', 'Fare', 'SibSp', 'Parch']], hue='Survived',height=3, aspect=0.75,diag_kind="hist", dropna=True, plot_kws={'alpha': 0.2})

#Drop features that will not have statistical value for our model
titanic_bruno.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace = True)

#Fill na values in the 'Embarked' feature with the forward value
titanic_bruno['Embarked'].fillna(method='ffill', inplace=True)

titanic_bruno.Embarked.isna().sum()

#Convert 'Sex' and 'Embarked' to Numerical
dummies = pd.get_dummies(titanic_bruno[['Sex', 'Embarked']], drop_first=True)

dummies.head()

#Concatenate the dummies variables to the titanic data frame
titanic_b = pd.concat([titanic_bruno, dummies], axis = 1)

#Drop redundant features
titanic_b.drop(['Sex', 'Embarked'], axis = 1, inplace = True)

titanic_b.head()

#Inspect 'Age' for NAN
titanic_b['Age'].isna().sum()

titanic_b['Age'].fillna(titanic_b['Age'].mean(), inplace = True)

titanic_b.describe()
titanic_b.info()

#Convert data types to float
titanic_b = titanic_b.astype(float)

titanic_b.info()


#Function to normalize data
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        
    return df_norm

#Applying normalization on the dataset
titanic_bruno_norm = min_max_scaling(titanic_b)

#Displaying the first two records
titanic_bruno_norm.head(2)

#Plot a histogram of the variables
titanic_bruno_norm.hist(figsize=(9,10), bins=10)

#Split the data set into independent and dependet variables
X_bruno = titanic_bruno_norm.drop('Survived', axis = 1)
y_bruno = titanic_bruno_norm['Survived']


#Use Train_Test_Split to randomly split the dataset into train and test datasets
X_train_bruno, X_test_bruno, y_train_bruno, y_test_bruno = train_test_split(
    X_bruno, y_bruno, test_size=0.3, random_state=98)

#Instantiate and train a Logistic model
bruno_model = LogisticRegression()
bruno_model.fit(X_train_bruno, y_train_bruno)

#Print the coefficients of all predictors
print(pd.DataFrame(zip(X_train_bruno.columns, np.transpose(bruno_model.coef_)), columns=['Predictor', 'Coefficients']))

#Use cross validation
scores = cross_val_score(bruno_model, X_train_bruno, y_train_bruno, cv=10)
print(scores)


#Running cross-validation with different splits
mean_score = []

for i in np.arange(0.10, 0.55, 0.05):
    X_train, X_test, y_train, y_test = train_test_split(
    X_bruno, y_bruno, test_size=i, random_state=98)
    
    bruno_model.fit(X_train, y_train)
    scores = cross_val_score(bruno_model, X_train, y_train, cv=10)
    mean_score.append(scores.mean())
    print(f"Split {i}\nMinimum Accuracy: {scores.min()}\nMean Accuracy: {scores.mean()}\nMaximum Accuracy: {scores.max()}\n ")
    

#Plot the mean scores for the cross-validation ran with different splits
plt.figure(figsize=(10,6))
plt.plot(np.arange(0.10, 0.55,0.05),mean_score,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Mean Score vs. Split')
plt.xlabel('Split')
plt.ylabel('Mean Score')

#Split the data set into 70% train and 30% test
X_train_bruno, X_test_bruno, y_train_bruno, y_test_bruno = train_test_split(
    X_bruno, y_bruno, test_size=0.3, random_state=98)

#Retrain the model
bruno_model.fit(X_train_bruno, y_train_bruno)

#Calculating the probability of each prediction
y_pred_bruno = bruno_model.predict_proba(X_test_bruno)

#Print probabilities
print(y_pred_bruno[:,1])

#Set the threshold to 50% and determine the outputs for the labels
y_pred_bruno_flag = np.where(y_pred_bruno[:,1]>0.5,1,0)
print(y_pred_bruno_flag)

#Calculate the accuracy score
accuracy_score(y_test_bruno, y_pred_bruno_flag)

# Null accuracy: accuracy that could be achieved by always predicting the most frequent class
y_test_bruno.value_counts()
print(f'The accuracy that should be achieved by always predicting the most frequent class is: \n{max(y_test_bruno.mean(), 1-y_test_bruno.mean())}')

#Confusion Matrix
cf_matrix_log_bruno = confusion_matrix(y_test_bruno, y_pred_bruno_flag)
print(cf_matrix_log_bruno)

#Plot the confusion Matrix
class_names=[0,1]
fig, ax = plt.subplots(figsize=(10,6))
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(cf_matrix_log_bruno, annot=True,cmap="PuBu" ,fmt='g')
ax.set_ylim([0,2])
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix - Logistic Regression',fontsize=20)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#Plot the ROC curve and display the area under the curve
logit_roc_auc = roc_auc_score(y_test_bruno, y_pred_bruno[:,1])
fpr, tpr, thresholds = roc_curve(y_test_bruno, y_pred_bruno[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#Print classification report
print(classification_report(y_test_bruno, y_pred_bruno_flag))
type(y_test_bruno)

#Making predictions with the threshold set to 5%
y_pred_bruno_flag2 = np.where(y_pred_bruno[:,1]>0.75,1,0)
print(y_pred_bruno_flag2)

#Calculate the accuracy score with the threshold set to 75%
accuracy_score(y_test_bruno, y_pred_bruno_flag2)

#Confusion Matrix
cf_matrix = confusion_matrix(y_test_bruno, y_pred_bruno_flag2)
print(cf_matrix)

#Classification report
print(classification_report(y_test_bruno, y_pred_bruno_flag2))


X_train_bruno.info()

#Making predictions with the training dataset
y_pred_bruno_train = bruno_model.predict_proba(X_train_bruno)
y_pred_bruno_flag_train = np.where(y_pred_bruno_train[:,1]>0.75,1,0)
print(y_pred_bruno_flag_train)

#Calculating the accuracy score
accuracy_score(y_train_bruno, y_pred_bruno_flag_train)

#Confusion Matrix
cf = confusion_matrix(y_train_bruno, y_pred_bruno_flag_train)
print(cf)

#Print the classification report
print(classification_report(y_train_bruno, y_pred_bruno_flag_train))

'''precision means what percentage of the positive predictions made were actually correct.

TP/(TP+FP)

'''

'''Recall in simple terms means, what percentage of actual positive predictions were correctly classified by the classifier.

TP/(TP+FN)

'''

'''F1 score can also be described as the harmonic mean or weighted average of precision and recall.

2x((precision x recall) / (precision + recall))

'''



