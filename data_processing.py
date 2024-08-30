import matplotlib
import numpy as py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Daten einlesen

train_titanic_file = pd.read_csv('../data/train.csv')

# Daten verarbeiten

train_titanic_file.drop('Cabin', axis = 1, inplace=True)

def preprocessing(df):
    MEDIAN_CLASS_1 = df[df["Pclass"]==1]["Age"].median()
    MEDIAN_CLASS_2 = df[df["Pclass"]==2]["Age"].median()
    MEDIAN_CLASS_3 = df[df["Pclass"]==3]["Age"].median()

    def changeAge(para):
        age = para[0]
        booking_class = para[1]

        if pd.isnull(age):
            if booking_class == 1:
                return MEDIAN_CLASS_1
            elif booking_class == 2:    
                 return MEDIAN_CLASS_2
            else:
                return MEDIAN_CLASS_3
        else: 
            return age
        
    train_titanic_file['Age'] = train_titanic_file[['Age','Pclass']].apply(changeAge, axis = 1)

preprocessing(train_titanic_file)
train_titanic_file.dropna(inplace=True)

gender = pd.get_dummies(train_titanic_file['Sex'], drop_first = True)
hafen = pd.get_dummies(train_titanic_file['Embarked'], drop_first = True)
train_new = pd.concat([train_titanic_file, gender,hafen], axis = 1)

train_new.drop(['Sex','Embarked','Name','Ticket','PassengerId'], axis = 1, inplace = True)

train_new.to_csv("data_for_model.csv")