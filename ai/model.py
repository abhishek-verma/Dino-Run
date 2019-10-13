from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from .naive_bayes import NaiveBayes


import pandas as pd
import numpy as np
from scipy.stats import mode

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random

np.seterr(divide='ignore', invalid='ignore')

classifier = 'our'
use_all = True

class Model:

    def __init__(self):
        #Create a Gaussian Classifier
        self.our = NaiveBayes()
        self.gnb = GaussianNB()
        self.knc = KNeighborsClassifier(n_neighbors=3)
        # self.clf = svm.SVC(gamma='scale') # NOT WORKING
        print("model initiallized")
        
        self.gen = 0

    def train(self, data):
        df = pd.DataFrame(data)

        self.gen = self.gen + 1
        print("gen: ", self.gen)
        if(self.gen % 5 == 0):
            self.plot(df)

        if(classifier == 'our' or use_all) :
            self.our.fit(df[['cod', 'coa', 'speed']].values, df['action'])
        # if(classifier == 'clf') :
        #     self.clf.fit(df[['cod', 'coa', 'speed']], df['action'])
        if(classifier == 'gnb' or use_all) :
            self.gnb.fit(df[['cod', 'coa', 'speed']], df['action'])
        if(classifier == 'knc' or use_all) :
            self.knc.fit(df[['cod', 'coa', 'speed']], df['action'])

        print("trained for data: \n", df[['cod', 'coa', 'speed', 'action']])

    def predict(self, cod, coa, speed):

        # print('predicting action... ')

        try :
            predicted_action = []
            if(classifier == 'our' or use_all) :
                predicted_action.append(self.our.predict([[cod, coa, speed]])[0])
            # if(classifier == 'clf') :
            #     predicted_action = self.clf.predict([[cod, coa, speed]])
            if(classifier == 'gnb' or use_all) :
                predicted_action.append(self.gnb.predict([[cod, coa, speed]])[0])
            if(classifier == 'knc' or use_all) :
                predicted_action.append(self.knc.predict([[cod, coa, speed]])[0])
            print('predicted action: ', predicted_action)

            action = []
            action.append(0)
            action.append(1)

            for a in predicted_action:
                # action[a] = action[a] + 1
                if(a == 1):
                    return 1

            return 0

            # if action[0] > action[1] :
            #     return 0
            # else: 
            #     return 1

        except Exception as e: 
            # print(e)
            return 0  

    def plot(self, df): 
        fig = pyplot.figure()
        ax = Axes3D(fig)

        cod = df['cod']
        coa = df['coa']
        speed = df['speed']
        action = df['action']
        color= ['red' if a == 0 else 'green' for a in action]
        # plt.scatter(arr1, arr2, color=color)
        # plt.show()
        ax.scatter(cod, speed, coa, color=color)

        ax.set_xlabel('COD')
        ax.set_ylabel('SPEED')
        ax.set_zlabel('COA')

        pyplot.show()
        