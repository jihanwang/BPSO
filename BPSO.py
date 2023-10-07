import random
from random import random
from random import random,randint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

class BPSO:
    def __init__(self, f_count, df):
        
        self.f_count  = f_count
        self.pos_act  = []
        self.position = []
        self.velocity = []
        self.pos_best = []
        self.y_actual = []
        self.y_predict= []
        self.fit_best = (-1, -1, -1)
        self.fitness  = (-1, -1, -1)
        self.df       = df.copy()
        
        self.initialize(f_count)
    
    # initialize 
    def initialize(self, f_count):
        self.f_count = f_count
        self.initalize_position(f_count)
        self.initialize_velocity(f_count)
    
    def set_data(self,data):
        self.df = data.copy()
        print(self.df.head())
        
    #Initialize the positions > 0.5  is set as 1
    def initalize_position(self,f_count):
        self.pos_act = np.random.uniform(low=0, high=1, size=f_count).tolist()
        self.position = [1 if po > 0.5 else 0  for po in self.pos_act]
        
    def initialize_velocity(self, f_count):
        self.velocity = np.random.uniform(low=-1, high=1, size=f_count).tolist()
        
    
    def drop_columns(self, X):
        print(X.shape)
        print(self.position)
        for index, value in enumerate(self.position):
            if value == 0 :
                X_1 = X.drop(X.columns[index], axis = 1)
        return X_1
    
    def classification_accuracy(self,y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                FP += 1
            if y_actual[i]==y_hat[i]==0:
                TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                FN += 1
        
        
        class_acc = float((TP+TN)) / float((TP+FP+TN+FN))
        
        if TP == 0 and FN == 0 :
            recall = 0
        else:
            recall  = float(TP) / float(TP + FN)
        
        if TP == 0 and FP == 0:
            precision = 0
        else:
            precision = float(TP) / float( TP + FP )         
        
        return (class_acc, recall, precision)
        
    
    def process_data(self):
                
        # Separate labels and features
        X = self.df.drop(columns=['class'])
        y = self.df['class']
        
        # X = self.drop_columns(X)
        

        # Convert the M to 1 and B to 0
        label = LabelEncoder()
        y = label.fit_transform(y)

        # Spilt the train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # we used 30% test data
        # check the size before beginning
        X_train.shape, X_test.shape, y_train.shape, y_test.shape
        
        # Logistic Regression
        LR = LogisticRegression()
        LR.fit(X_train, y_train)
        LR.score(X_train, y_train)
        y_pred = LR.predict(X_test)
        y_pred_train = LR.predict(X_train)
        
        # find accuracy
        ac = accuracy_score(y_test, y_pred)
        ac_train = accuracy_score(y_train, y_pred_train)
        # Code for ROC_AUC curve
        rc = roc_auc_score(y_test, y_pred)
        
        class_acc = self.classification_accuracy(y_test, y_pred)
        
        self.y_actual = y_test
        self.y_predict = y_pred
         
        return class_acc
    
    # fitness check, checks accuarcy and precision and accurarcy 
    def fitness_check(self,fitness, fit_best):
        is_fitness = False
        
        if fitness[0] > fit_best[0] or fit_best[0] == -1:
            if fitness[1] >= fit_best[1] and fitness[2] >= fit_best[2]:
                is_fitness = True
        
        return is_fitness

    #evaluate the fitness
    def evaluate_fitness(self):
        self.fitness = self.process_data()
        
        if  self.fitness_check(self.fitness, self.fit_best):
            self.pos_best  = self.position.copy()
            self.fit_best = self.fitness
    
    def update_velocity(self, pos_best_global):
        
        c1 = 1
        c2 = 2
        w  = 0.5

        for i in range(0, self.f_count):

            r1 = np.random.uniform(low=-1, high=1, size=1)[0]#random()
            r2 = np.random.uniform(low=-1, high=1, size=1)[0]#random()

            velocity_cog = c1*r1*(self.pos_best[i]-self.position[i])
            velocity_soc = c2*r2*(pos_best_global[i]-self.position[i])
            
            self.velocity[i]=w*self.velocity[i]+velocity_cog+velocity_soc
    
    def update_position(self):
        
        for i in range(0, self.f_count):
            self.pos_act[i] = self.pos_act[i] + self.velocity[i]
            
            
            if self.pos_act[i] > 1:
                self.pos_act[i] = 0.9
            
            if self.pos_act[i] < 0 :
                self.pos_act[i] = 0.0
                
            self.position[i] = 1 if self.pos_act[i] > 0.5 else 0       
        
        
    def print_position(self):
        print(self.position)
    
    def print_velocity(self):
        print(self.velocity)
    
def pso_calculate(f_count, df):
    y_actual = []
    y_predict = []
    fitness_best_g = (-1, -1, -1)
    pos_fitness_g = []
    swarm = []
    no_population = 500
    
    for i in range(0,no_population):
        swarm.append(BPSO(f_count, df))
    
    #optimize 
    index = 0
    
    while index < 50:
        
        for pos in range(0, no_population):
            swarm[pos].evaluate_fitness()
            
            if swarm[pos].fitness_check(swarm[pos].fitness, fitness_best_g): 
                pos_fitness_g = list(swarm[pos].position)
                fitness_best_g = (swarm[pos].fitness)

                y_actual = swarm[pos].y_actual
                y_predict = swarm[pos].y_predict
            
        for pos in range(0, no_population):
            swarm[pos].update_velocity(pos_fitness_g)
            swarm[pos].update_position()
        
        print('index',index)
        index+=1
    
    print('\n Final Solution:')
    print(pos_fitness_g)
    print(fitness_best_g)
    cm_2 = confusion_matrix(y_actual, y_predict)


df = pd.read_csv('E:/Plos One/code/Lung_Cancer_19_proteins_snr_gt0.4.csv')
pso_calculate(19,df)
