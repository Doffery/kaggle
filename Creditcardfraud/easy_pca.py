#!/usr/bin/python
#!encoding=utf-8

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    sdataset = pd.read_csv('./data/creditcard.csv')
    #print sdataset.head()
    train = sdataset.ix[:, sdataset.columns != 'Class']
    target = sdataset['Class']
    print sdataset['Time'][sdataset.Class==1].describe()
    plt.hist(sdataset['Time'][sdataset.Class==1], bins=30)
    plt.xlabel('Time')
    plt.ylabel('Class Count')
    plt.show()


if __name__ == "__main__":
    main()
