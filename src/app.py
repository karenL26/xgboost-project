###################################
# XGB Classifier Project          #
###################################
### Load libraries and modules ###
# Dataframes and matrices ----------------------------------------------
import pandas as pd
import numpy as np
import os
import joblib
# Machine learning -----------------------------------------------------
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
# Preprocessing --------------------------------------------------------
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Metrics --------------------------------------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error

######################
# Data Preprocessing #
######################
# Loading the dataset
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv')
# Create a copy of the original dataset
df = df_raw.copy()

# Fill the missing values in Age with the mean
df['Age']=df['Age'].fillna(df['Age'].mean())
# Fill the missing values in Embarked with the mode
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
# Dropping data with feature above upper bound
df.drop(df[(df['Fare'] > 66)].index, inplace=True)
df.drop(df[(df['SibSp'] > 2)].index, inplace=True)
df.drop(df[(df['Age'] > 65)].index, inplace=True)
# Merge SibSp and Parch in new column to indicate number of family members on board
df["realtives"] = df["SibSp"] + df["Parch"]
# Encoding the 'Sex' column
df['Sex'] = df['Sex'].map({'male' : 0, 'female': 1})
# Encoding the 'Embarked' column
df['Embarked'] = df['Embarked'].map({'S' : 0, 'C': 1, 'Q': 2})
# Drop irrelevant columns as PassengerId, Name, Ticket, Cabin
df=df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'])

#####################
# Model and results #
#####################
X_inb = df.drop('Survived',axis=1)
y_inb = df['Survived']
# We use random Over-Sampling to add more copies to the minority class
ros =  RandomOverSampler()
X,y = ros.fit_resample(X_inb,y_inb)
# Spliting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=40)

pipeline = make_pipeline(StandardScaler(), XGBClassifier(colsample_bytree = 0.5, eta= 0.05, gamma= 0.0, max_depth= 15, min_child_weight= 1, n_estimators= 100))
pipeline.fit(X_train, y_train)
y_pred=pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
print("Confusion Matrix\n",cm)
print(classification_report(y_test,y_pred))
print(mean_absolute_error(y_test, y_pred))
print("Score in train dataset:", round(pipeline.score(X_train, y_train), 4))
print("Score in test dataset:", round(pipeline.score(X_test, y_test), 4)) 

# We save the model with joblib
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../models/gbx.pkl')

joblib.dump(pipeline, filename)