import itertools
import math
import os
import numpy as np
import pandas as pd
import random
import warnings
from sklearn.model_selection  import train_test_split
from sklearn.metrics import confusion_matrix as cnfm
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import wandb
from keras.datasets import fashion_mnist
from keras.datasets import mnist
class Neural:
    def __init__(self,Size_of_Input, Number_of_Neuron_each_Layer, Number_of_Layers, activation_function, typeOfInit, L2reg_const = 0):
        self.activation_function = activation_function
        self.Size_of_Input = Size_of_Input
        self.Number_of_Layers = Number_of_Layers
        self.Number_of_Neuron_each_Layer = Number_of_Neuron_each_Layer
        self.L2reg_const = L2reg_const
        self.W,self.b = self.initializer(typeOfInit)

    
    def initializer(self, init):        
        W = []
        b = []
        if init == 'random':
            input_neuron = self.Number_of_Neuron_each_Layer[0]
            input_size = self.Size_of_Input
            W.append(np.random.randn(input_neuron, input_size))
            number_of_layers = self.Number_of_Layers
            for i in range(1,number_of_layers):
                curr_layer = self.Number_of_Neuron_each_Layer[i]
                prev_layer = self.Number_of_Neuron_each_Layer[i-1]
                W.append(np.random.randn(curr_layer,prev_layer))

            for i in range( number_of_layers):
                curr_layer = self.Number_of_Neuron_each_Layer[i]
                b.append(np.random.rand(curr_layer))
        if (init == 'xavier'):
            input_neuron = self.Number_of_Neuron_each_Layer[0]
            parameter =  input_neuron+ self.Size_of_Input
            input_size = self.Size_of_Input
            number_of_layers = self.Number_of_Layers
            W.append(np.random.normal(0,math.sqrt(2/(parameter)), ( input_neuron, input_size)))
            for i in range(1,self.Number_of_Layers):
                W.append(np.random.normal(0, math.sqrt(2/(self.Number_of_Neuron_each_Layer[i]+self.Number_of_Neuron_each_Layer[i-1])),(self.Number_of_Neuron_each_Layer[i],self.Number_of_Neuron_each_Layer[i-1])))

            for i in range(number_of_layers):
                curr_layer= self.Number_of_Neuron_each_Layer[i]
                b.append(np.random.rand(curr_layer))
        return W,b


    def activation(self, Z):
        if self.activation_function == 'ReLU':
            res = self.ReLU(Z)
            return res
        elif self.activation_function == 'tanh':
            res =  self.tanh(Z)
            return res
        elif self.activation_function == 'sigmoid':
            res = self.sigmoid(Z)
            return res


    def activation_derivative(self,Z):
        if self.activation_function == 'ReLU':
            res = self.ReLU_derivative(Z)
            return res
        if self.activation_function == 'tanh':
            res = self.tanh_derivative(Z)
            return res
        if self.activation_function == 'sigmoid':
            res = self.sigmoid_derivative(Z)
            return res
    def ReLU(self,Z):
        res= np.maximum(0,Z)
        return res

    def ReLU_derivative(self,Z):
        res = [1 if x>0 else 0 for x in Z]
        return res

    def tanh(self, Z):
        res = np.array([((np.exp(x) - np.exp(-x))/((np.exp(x) + np.exp(-x)))) for x in Z])
        return res
                 
    def tanh_derivative(self, Z):
        res = np.array(1 - self.tanh(Z)**2)
        return res
                 
    def sigmoid_derivative(self,Z):
        res = self.sigmoid(Z)*(1-self.sigmoid(Z))
        return res

    def sigmoid(self,x):
        warnings.filterwarnings('ignore')
        res = np.where(x>=0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))
        return res
    
    def softmax_function(self,Z):
            maxZ=Z.max()
            Z =   Z - maxZ# This is done to normalize the dataset
            prob =np.exp(Z)
            sumprob=np.sum(np.exp(Z),axis=0)
            return (prob/sumprob)

    def forward_propagation(self,Input):
        Input = np.array(Input)
        A = []
        H = []
        res = self.W[0].dot(Input) + self.b[0]
        A.append(res)
        number_of_layers =  self.Number_of_Layers
        for i in range(1, number_of_layers):
            H.append(self.activation(A[-1]))
            preactivation = self.W[i].dot(H[-1]) + self.b[i]
            A.append(preactivation)
        y_hat = self.softmax_function(A[-1])
        return A, H, y_hat

    def backward_propagation(self, A, H, y_hat, y, Input, loss_type):
        Input = np.array(Input)
        H.insert(0,Input)
        delW = []
        delb = []
        delA = []
        delH = []

        last_layer=self.Number_of_Neuron_each_Layer[-1]
        zeros= np.zeros(last_layer)
        ey = zeros
        ey[y] = 1
        
        if loss_type=="squared_error":
            der1=(y_hat - ey)
            der2=(y_hat - y_hat**2)
            delA.append(np.array(der1*der2))
        else:
        # delA and delH have reverse indexing
            res=-(ey - y_hat)
            delA.append(np.array(res))
        number_of_layers= self.Number_of_Layers
        for i in range( number_of_layers-1,-1,-1):
            delastAshape=delA[-1].shape[0]
            hlastshape=H[i].shape[0]
            regulariz=self.L2reg_const*self.W[i]
            delW.insert(0,delA[-1].reshape(delastAshape,1).dot(H[i].reshape(hlastshape,1).T) +regulariz )
            delb.insert(0,delA[-1])
            temp=self.W[i].T.dot(delA[-1])
            delH.append(temp)
            if i-1>=0:
                delA.append(np.multiply(delH[-1], self.activation_derivative(A[i-1])))
        return delW,delb
    
    
    def initialize(self, Size_of_Input,Number_of_Layers,Number_of_Neuron_each_Layer):
        W, b = [], []
        each_layer_neuron = Number_of_Neuron_each_Layer[0]
        input_size = Size_of_Input
        W.append(np.zeros(( Number_of_Neuron_each_Layer[0],input_size )))
        for i in range(1,Number_of_Layers):
            curr_layer = Number_of_Neuron_each_Layer[i]
            prev_layer = Number_of_Neuron_each_Layer[i-1]
            W.append(np.zeros((curr_layer,prev_layer)))
        for i in range(Number_of_Layers):
            curr_layer = Number_of_Neuron_each_Layer[i]
            b.append(np.zeros(curr_layer))            
        return W, b

    
    def optimize(self, X, Y, val_images,val_labels,optimizer, learning_rate, max_epochs,batch_size,loss_type,momentum=0.5,beta = 0.89, epsilon = 1e-6,beta1 = 0.89,beta2 = 0.989):
        if optimizer == 'sgd':
          self.stochastic_gradient_descent(X, Y, val_images,val_labels, learning_rate, max_epochs,loss_type)
        if optimizer == 'momentum':
          self.momentum_gradient_descent(X, Y, val_images,val_labels, learning_rate, max_epochs,batch_size,loss_type,momentum)
        if optimizer == 'nag':
          self.nesterov_accelerated_gradient_descent(X, Y, val_images,val_labels, learning_rate, max_epochs,batch_size,loss_type,momentum)
        if optimizer == 'rmsprop':
          self.rmsprop(X, Y, val_images,val_labels, learning_rate, max_epochs,batch_size,loss_type,beta, epsilon)
        if optimizer == 'adam':
          self.adam(X, Y, val_images,val_labels, learning_rate, max_epochs,batch_size,loss_type,beta1,beta2,epsilon)
        if optimizer == 'nadam':
          self.nadam(X, Y, val_images,val_labels, learning_rate, max_epochs,batch_size,loss_type,beta1,beta2,epsilon)


    def stochastic_gradient_descent(self,X, Y, val_images,val_labels, learning_rate, max_epochs,loss_type):
        number_of_inputs = self.Size_of_Input
        layers_size = self.Number_of_Layers
        neurons = self.Number_of_Neuron_each_Layer
        for j in range(max_epochs):
            correct = 0
            error = 0
            delW, delb = self.initialize(number_of_inputs,layers_size,neurons)
            xsize=X.shape[0]
            for i in range(xsize):
                features=X[i]
                A,H,y_hat = self.forward_propagation(features)
                upW=self.W
                s = [x.sum() for x in upW]
                zeros=np.zeros(self.Number_of_Neuron_each_Layer[-1])
                ey = zeros
                ey[Y[i]] = 1
                if loss_type == "squared_error":
                      sqrd = 0.5*np.sum((ey-y_hat)**2)
                      reg =  self.L2reg_const/2*sum(s)
                      error += sqrd + reg 
                else:
                      cross =-math.log(y_hat[Y[i]])
                      reg = self.L2reg_const/2*sum(s)
                      error += cross + reg

                delW,delb = self.backward_propagation(A,H,y_hat,Y[i],X[i],loss_type)
                res=np.argmax(y_hat)
                if(res == Y[i]):
                    correct +=1
                
                for i in range(self.Number_of_Layers):
                    lrdelw = learning_rate*delW[i]
                    self.W[i] = self.W[i] - lrdelw
                    lrdelb = learning_rate*delb[i]
                    self.b[i] = self.b[i] - lrdelb

            v_error, v_accruracy = self.val_loss_and_accuracy(val_images, val_labels,loss_type)
            sizex=X.shape[0]
            error =error/sizex
            sizexacc= X.shape[0]
            accuracy = correct/sizexacc*100
            wandb.log({'epoch' : j, 'train_loss' : error, 'train_accuracy' : accuracy,'valid_loss' : v_error,'valid_accuracy' : v_accruracy})


    def momentum_gradient_descent(self,X, Y, val_images,val_labels, learning_rate, max_epochs,batch_size, loss_type,gamma = 0.6):
        number_of_input=self.Size_of_Input
        number_of_layers=self.Number_of_Layers
        neuron=self.Number_of_Neuron_each_Layer
        updateW, updateb = self.initialize(number_of_input,number_of_layers,neuron)

        for j in range(max_epochs):
            correct = 0
            error = 0

            delW, delb = self.initialize(number_of_input, number_of_layers,neuron)
            xsize=X.shape[0]
            
            for i in range(X.shape[0]):
                features=X[i]
                A,H,y_hat = self.forward_propagation(features)
                upW=self.W
                s = [x.sum() for x in upW]
                ey = np.zeros(self.Number_of_Neuron_each_Layer[-1])
                ey[Y[i]] = 1
                if loss_type == "squared_error":
                      sqrd = 0.5*np.sum((ey-y_hat)**2)
                      reg =  self.L2reg_const/2*sum(s)
                      error += sqrd + reg 
                else : 
                     cross = -math.log(y_hat[Y[i]])
                     reg = self.L2reg_const/2*sum(s)
                     error += cross + reg
                
                w,b = self.backward_propagation(A,H,y_hat,Y[i],X[i],loss_type)

                for k in range( number_of_layers):
                    prevdelW = delW[k]
                    delW[k]  = prevdelW + w[k]
                    prevdelB = delb[k]
                    delb[k]  = prevdelB + b[k]

                for k in range( number_of_layers):
                    gammaupW = gamma*updateW[k]
                    lrdelw = learning_rate*delW[k]
                    updateW[k] = gammaupW + learning_rate*delW[k]  

                    gammaupB = gamma*updateb[k]
                    lrdelb = learning_rate*delb[k]
                    updateb[k] = gammaupB + lrdelb
                temp=i%batch_size
                if  (temp == 0 and i!=0) or i==xsize-1:
                    delW, delb = self.initialize(number_of_input, number_of_layers,neuron)
                    for k in range( number_of_layers):
                        prevW1 = self.W[k] 
                        self.W[k] = prevW1 -updateW[k]
                        prevB1 = self.b[k]  
                        self.b[k] = prevB1 -updateb[k]
                res = np.argmax(y_hat)
                if(res == Y[i]):
                    correct +=1

                
            v_error, v_accruracy = self.val_loss_and_accuracy(val_images, val_labels,loss_type)
            sizex=X.shape[0]
            error =error/sizex
            sizexacc= X.shape[0]
            accuracy = correct/sizexacc*100
            wandb.log({'epoch' : j, 'train_loss' : error, 'train_accuracy' : accuracy,'valid_loss' : v_error,'valid_accuracy' : v_accruracy})


    def nesterov_accelerated_gradient_descent(self, X, Y, val_images,val_labels, learning_rate, max_epochs,batch_size,loss_type ,gamma = 0.5):
        number_of_input= self.Size_of_Input
        layers_size=self.Number_of_Layers
        neurons=self.Number_of_Neuron_each_Layer
        updateW, updateb = self.initialize( number_of_input, layers_size,neurons)
        lookaheadW, lookaheadb = self.initialize( number_of_input,  layers_size,neurons)
        thetaW, thetab = self.initialize( number_of_input, layers_size,neurons)

        for j in range(max_epochs):
            correct = 0
            error = 0

            delW, delb = self.initialize(number_of_input,layers_size,neurons)
                
            for k in range(layers_size):
                thetaW[k] = self.W[k]
                thetab[k] = self.b[k]

            for k in range(layers_size):
                gammaupW=gamma*updateW[k]
                lookaheadW[k] = thetaW[k] - gammaupW 
                gammaupB=gamma*updateb[k]
                lookaheadb[k] = thetab[k] - gammaupB
                self.W[k] = lookaheadW[k]
                self.b[k] = lookaheadb[k]

            xsize=X.shape[0]
            for i in range(xsize):
                A,H,y_hat = self.forward_propagation(X[i])
                Wup=self.W
                s = [x.sum() for x in Wup ]
               # cross=-math.log(y_hat[Y[i]])
               # regu= self.L2reg_const/2*sum(s)
                #error += cross + regu
                ey = np.zeros(self.Number_of_Neuron_each_Layer[-1])
                ey[Y[i]] = 1
                if loss_type == "squared_error":
                      sqrd = 0.5*np.sum((ey-y_hat)**2)
                      reg =  self.L2reg_const/2*sum(s)
                      error += sqrd + reg 
                else : 
                    cross = -math.log(y_hat[Y[i]])
                    reg = self.L2reg_const/2*sum(s)
                    error += cross + reg
                w,b = self.backward_propagation(A,H,y_hat,Y[i],X[i],loss_type)

                for k in range(layers_size):
                    prevdW=delW[k]
                    delW[k] =prevdW + w[k]
                    prevdB=delb[k]
                    delb[k] =prevdB + b[k]

                for k in range( layers_size):
                    gammaW = gamma*updateW[k]
                    lrdelW= learning_rate*delW[k] 
                    updateW[k] =  gammaW + lrdelW

                    gammab =   gamma*updateb[k]
                    lrdelb=    learning_rate*delb[k]
                    updateb[k] = gammab +  lrdelb

                temp=i%batch_size
                if  (temp == 0 and i!=0) or i==xsize-1:
                    delW, delb = self.initialize(self.Size_of_Input,layers_size,self.Number_of_Neuron_each_Layer)
                    for k in range(layers_size):
                        befW = self.W[k]
                        self.W[k] = befW -updateW[k] 
                        befB = self.b[k]
                        self.b[k] =befB -updateb[k]
                res=np.argmax(y_hat)
                if(res == Y[i]):
                    correct +=1
            
           # error /= X.shape[0]
           # accuracy = correct/X.shape[0]*100
           # v_error, v_accruracy = self.val_loss_and_accuracy(val_images, val_labels)
            v_error, v_accruracy = self.val_loss_and_accuracy(val_images, val_labels,loss_type)
            sizex=X.shape[0]
            error =error/sizex
            sizexacc= X.shape[0]
            accuracy = correct/sizexacc*100
            wandb.log({'epoch' : j, 'train_loss' : error, 'train_accuracy' : accuracy,'valid_loss' : v_error,'valid_accuracy' : v_accruracy})



    def rmsprop(self,X, Y, val_images,val_labels, learning_rate, max_epochs,batch_size,loss_type, beta = 0.89, epsilon = 1e-6):
        number_of_inputs=self.Size_of_Input
        layers_size=self.Number_of_Layers
        neurons=self.Number_of_Neuron_each_Layer
        v_W, v_b = self.initialize(number_of_inputs,layers_size,neurons)

        for j in range(max_epochs):
            error = 0
            correct = 0
            xsize=X.shape[0]

            delW, delb = self.initialize(number_of_inputs,layers_size,neurons)
    
            for i in range(xsize):
                val= X[i]
                A,H,y_hat = self.forward_propagation(val)
                Wup=self.W
                s = [x.sum() for x in Wup ]
                #cross= -math.log(y_hat[Y[i]])
               # reg= self.L2reg_const/2*sum(s)
                #error += cross  + reg
                ey = np.zeros(self.Number_of_Neuron_each_Layer[-1])
                ey[Y[i]] = 1
                if loss_type == "squared_error":
                      sqrd = 0.5*np.sum((ey-y_hat)**2)
                      reg =  self.L2reg_const/2*sum(s)
                      error += sqrd + reg 
                else : 
                    cross = -math.log(y_hat[Y[i]])
                    reg = self.L2reg_const/2*sum(s)
                    error += cross + reg

                w,b = self.backward_propagation(A,H,y_hat,Y[i],X[i],loss_type)

                for k in range(layers_size):
                    prevdelW = delW[k]
                    delW[k]  = prevdelW + w[k]
                    prevdelB = delb[k]
                    delb[k]  = prevdelB +  b[k]
                res=np.argmax(y_hat)
                if(res == Y[i]):
                    correct +=1

                for k in range(layers_size):
                    betavw1=beta*v_W[k]
                    betavw2=(1-beta)*delW[k]**2 
                    v_W[k] = betavw1  + betavw2 

                    betavb1=  beta*v_b[k]
                    betavb2=  (1-beta)*delb[k]**2 
                    v_b[k] = betavb1 + betavb2
         
                temp= i%batch_size
                if  (temp== 0 and i!=0) or i==xsize-1:
                    for k in range(layers_size):
                        betavw1= beta*v_W[k]
                        betavw2= (1-beta)*delW[k]**2  
                        v_W[k] =  betavw1 +  betavw2

                        betavb1=  beta*v_b[k]   
                        betavb2=  (1-beta)*delb[k]**2
                        v_b[k] = betavb1 + betavb2
                    for k in range(layers_size):
                        lrdelw=(learning_rate*delW[k])
                        sqrtvW=np.sqrt(v_W[k] + epsilon)
                        self.W[k] = self.W[k] - lrdelw/sqrtvW

                        lrdelb=(learning_rate*delb[k])
                        sqrtvB=np.sqrt(v_b[k] + epsilon)
                        self.b[k] = self.b[k] - lrdelb/sqrtvB
                    delW, delb = self.initialize(number_of_inputs,layers_size,neurons) 
            

            v_error, v_accruracy = self.val_loss_and_accuracy(val_images, val_labels,loss_type)
            sizex=X.shape[0]
            error =error/sizex
            sizexacc= X.shape[0]
            accuracy = correct/sizexacc*100
            wandb.log({'epoch' : j, 'train_loss' : error, 'train_accuracy' : accuracy,'valid_loss' : v_error,'valid_accuracy' : v_accruracy})


    
    def adam(self,X, Y, val_images,val_labels, learning_rate, max_epochs,batch_size, loss_type,beta1 = 0.89,beta2 = 0.989,epsilon = 1e-8):
        number_of_input = self.Size_of_Input
        number_of_layers= self.Number_of_Layers
        neurons= self.Number_of_Neuron_each_Layer
        m_W, m_b = self.initialize( number_of_input,number_of_layers,neurons)
        m_hat_W, m_hat_b = self.initialize(  number_of_input,number_of_layers,neurons)
        v_W, v_b = self.initialize( number_of_input,number_of_layers,neurons)
        v_hat_W, v_hat_b = self.initialize( number_of_input,number_of_layers,neurons)
        
        for j in range(0, max_epochs):
            correct = 0
            error = 0
            xsize = X.shape[0]
            delW, delb = self.initialize( number_of_input,number_of_layers,neurons)
            
            for i in range(xsize):
                features=X[i]
                A,H,y_hat = self.forward_propagation(features)
                upW = self.W
                s = [x.sum() for x in upW]
               # cross = -math.log(y_hat[Y[i]])
                #reg = self.L2reg_const/2*sum(s)
               # error += cross + reg
                ey = np.zeros(self.Number_of_Neuron_each_Layer[-1])
                ey[Y[i]] = 1
                if loss_type == "squared_error":
                      sqrd = 0.5*np.sum((ey-y_hat)**2)
                      reg =  self.L2reg_const/2*sum(s)
                      error += sqrd + reg 
                else : 
                    cross = -math.log(y_hat[Y[i]])
                    reg = self.L2reg_const/2*sum(s)
                    error += cross + reg
               

                w,b = self.backward_propagation(A,H,y_hat,Y[i],X[i],loss_type)

                for k in range(number_of_layers):
                    prevdelW = delW[k]
                    delW[k] = prevdelW + w[k]
                    prevdelB = delb[k]
                    delb[k] = prevdelB + b[k]
                res = np.argmax(y_hat) 
                if(res == Y[i]):
                    correct +=1
                temp = i%batch_size
                if  (temp == 0 and i!=0) or i==xsize-1:
                    for k in range(number_of_layers):
                        betavW1 =  beta2*v_W[k]
                        betavW2 = (1-beta2)*delW[k]*delW[k]
                        v_W[k] =   betavW1 + betavW2

                        betavB1 = beta2*v_b[k]
                        betavB2=  (1-beta2)*delb[k]*delb[k]
                        v_b[k] =  betavB1 + betavB2

                        betamW1 = beta1*m_W[k]
                        betadelw =  (1-beta1)*delW[k]
                        m_W[k] = betamW1 + betadelw

                        betamb1 = beta1*m_b[k]
                        betadelb= beta1*m_b[k]
                        m_b[k] = betamb1 + betadelb

                        powbeta1= (math.pow(beta1,j))
                        m_hat_W[k] = m_W[k]/ powbeta1
                        m_hat_b[k] = m_b[k]/powbeta1

                        powbeta2= (math.pow(beta2,j))
                        v_hat_W[k] = v_W[k]/powbeta2
                        v_hat_b[k] = v_b[k]/powbeta2
                    
                    for k in range(number_of_layers):
                        lrW = learning_rate*m_hat_W[k]
                        sqrtvW = np.sqrt(v_hat_W[k] + epsilon) 
                        self.W[k] = self.W[k] - (lrW)/sqrtvW

                        lrb = learning_rate*m_hat_b[k]
                        sqrtvb = np.sqrt(v_hat_b[k] + epsilon)
                        self.b[k] = self.b[k] - (lrb)/sqrtvb
                    delW, delb = self.initialize(number_of_input,number_of_layers,neurons)
                                
            v_error, v_accruracy = self.val_loss_and_accuracy(val_images, val_labels,loss_type)
            sizex=X.shape[0]
            error =error/sizex
            sizexacc= X.shape[0]
            accuracy = correct/sizexacc*100
            wandb.log({'epoch' : j, 'train_loss' : error, 'train_accuracy' : accuracy,'valid_loss' : v_error,'valid_accuracy' : v_accruracy})
    
    def nadam(self, X, Y, val_images,val_labels, learning_rate, max_epochs,batch_size, loss_type,beta1 = 0.89,beta2 = 0.989,epsilon = 1e-8):
        number_of_input = self.Size_of_Input
        number_of_layers= self.Number_of_Layers
        neurons= self.Number_of_Neuron_each_Layer
        v_W, v_b = self.initialize( number_of_input,number_of_layers,neurons)
        v_hat_W, v_hat_b = self.initialize( number_of_input,number_of_layers,neurons)
        m_W, m_b = self.initialize( number_of_input,number_of_layers,neurons)
        m_hat_W, m_hat_b = self.initialize(  number_of_input,number_of_layers,neurons)
        
        for j in range(0, max_epochs):
            correct = 0
            error = 0
            xsize = X.shape[0]
            delW, delb = self.initialize( number_of_input,number_of_layers,neurons)
            
            for i in range(xsize):
                features=X[i]
                A,H,y_hat = self.forward_propagation(features)
                upW = self.W
                s = [x.sum() for x in upW]
               # cross = -math.log(y_hat[Y[i]])
                #reg = self.L2reg_const/2*sum(s)
               # error += cross + reg
                ey = np.zeros(self.Number_of_Neuron_each_Layer[-1])
                ey[Y[i]] = 1
                if loss_type == "squared_error":
                      sqrd = 0.5*np.sum((ey-y_hat)**2)
                      reg =  self.L2reg_const/2*sum(s)
                      error += sqrd + reg 
                else : 
                    try:
                        cross = -math.log(y_hat[Y[i]])
                        reg = self.L2reg_const/2*sum(s)
                        error += cross + reg
                    except ValueError:
                        error += 10000
               

                w,b = self.backward_propagation(A,H,y_hat,Y[i],X[i],loss_type)

                for k in range(number_of_layers):
                    prevdelW = delW[k]
                    delW[k] = prevdelW + w[k]
                    prevdelB = delb[k]
                    delb[k] = prevdelB + b[k]
                res = np.argmax(y_hat) 
                if(res == Y[i]):
                    correct +=1
                temp = i%batch_size
                if  (temp == 0 and i!=0) or i==xsize-1:
                    for k in range(number_of_layers):
                        betavW1= beta2*v_W[k]
                        betavW2= (1-beta2)*delW[k]**2
                        v_W[k] =  betavW1 + betavW2

                        betavB1=beta2*v_b[k]
                        betavB2=(1-beta2)*delb[k]**2
                        v_b[k] = betavB1  + betavB2 

                        betamW1 = beta1*m_W[k]
                        betadelw =  (1-beta1)*delW[k]
                        m_W[k] = betamW1 + betadelw

                        betamb1 = beta1*m_b[k]
                        betadelb= beta1*m_b[k]
                        m_b[k] = betamb1 + betadelb

                        powbeta1= (math.pow(beta1,j))
                        m_hat_W[k] = m_W[k]/ powbeta1
                        m_hat_b[k] = m_b[k]/powbeta1

                        powbeta2= (math.pow(beta2,j))
                        v_hat_W[k] = v_W[k]/powbeta2
                        v_hat_b[k] = v_b[k]/powbeta2
                    
                    for k in range(number_of_layers):
                        beta_mw=beta1*m_hat_W[k]
                        beta_dw=(1-beta1)*delW[k]/(1-beta1)
                        sqrt_dw=np.sqrt(v_hat_W[k] + epsilon)
                        self.W[k] = self.W[k] - (learning_rate*(beta_mw + beta_dw))/sqrt_dw

                        beta_mb=beta1*m_hat_b[k]
                        beta_db=(1-beta1)*delb[k]/(1-beta1)
                        sqrt_db=np.sqrt(v_hat_b[k] + epsilon)
                        self.b[k] = self.b[k] - (learning_rate*(beta_mb+beta_db))/sqrt_db
                    delW, delb = self.initialize(number_of_input,number_of_layers,neurons)
                                
            v_error, v_accruracy = self.val_loss_and_accuracy(val_images, val_labels,loss_type)
            sizex=X.shape[0]
            error =error/sizex
            sizexacc= X.shape[0]
            accuracy = correct/sizexacc*100
            wandb.log({'epoch' : j, 'train_loss' : error, 'train_accuracy' : accuracy,'valid_loss' : v_error,'valid_accuracy' : v_accruracy})
    def val_loss_and_accuracy(self,val_data,val_labels,loss_type):
        val_loss = []
        val_accuracy = []
        val_correct = 0
        val_error = 0

        for i in range(val_data.shape[0]):
            A,H,y_hat = self.forward_propagation(val_data[i]) 
            upW=self.W
            s = [x.sum() for x in upW]
            if loss_type == "squared_error":
                ey = np.zeros(self.Number_of_Neuron_each_Layer[-1])
                ey[val_labels[i]] = 1
                error=0.5*np.sum((ey-y_hat)**2)
                regerror= self.L2reg_const/2*sum(s)
                val_error += error + regerror
            else:
                 regu=self.L2reg_const/2*sum(s)
                 cross=-math.log(y_hat[val_labels[i]])
                 val_error += cross + regu
            maxi=np.argmax(y_hat)
            if  maxi == val_labels[i]:
                val_correct += 1
        m=val_data.shape[0]
        m1=val_data.shape[0]
        return val_error/m, val_correct/m1*100


    def test(self,test_data,test_labels):
        correct = 0
        y_hat = []
        testsize=test_data.shape[0]
        for i in range(testsize):

            A,H,y = self.forward_propagation(test_data[i])
            res=np.argmax(y)
            if res == test_labels[i]:
                correct += 1
            y_hat.append(y)
        a=np.argmax(np.array(y_hat),axis=1)
        b=correct/test_data.shape[0]*100
        return a,b