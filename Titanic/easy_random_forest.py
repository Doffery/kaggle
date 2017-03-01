#!/usr/bin/python
#!encoding=utf-8

from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import numpy as np
import pandas as pd

def main():
    #tdataset = genfromtxt(open('./data/train.csv', 'r'), delimiter=',', quotechar='"')[1:]
    tdataset = pd.read_csv('./data/train.csv')
    testdata = pd.read_csv('./data/test.csv')
    full_data = [tdataset, testdata]
    for dataset in full_data:
        dataset['Sex'] = dataset['Sex'].map({'female':0, 'male':1}).astype(int)
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
        dataset['Embarked'] = dataset['Embarked'].\
                map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        dataset['Fare'] = dataset['Fare'].\
                fillna(dataset['Fare'].median()).astype(int)
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        age_null_random_list = np.random.randint(age_avg - age_std, \
                age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)
    feature_now = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = tdataset['Survived']
    train = tdataset[feature_now]
    #testdata = genfromtxt(open('./data/test.csv', 'r'), delimiter=',', quotechar='"')[1:]
    test = testdata[feature_now]

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)
    result = testdata.loc[:,['PassengerId']].assign(Survived=rf.predict(test))
    result.to_csv('./data/testsub.csv', index=False)
    #savetxt('./data/testsub.csv', rf.predict(test), fmt='%d',  delimiter=',')

if __name__ == "__main__":
    main()
