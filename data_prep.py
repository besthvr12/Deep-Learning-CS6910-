

import numpy as np
import pandas as pd
from sklearn.model_selection  import train_test_split
from keras.datasets import fashion_mnist, mnist

def data(use_dataset):
        if(use_dataset=="fashion_mnist"):
            (train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
            train_images, val_images, train_labels, val_labels  = train_test_split(train_images,train_labels,test_size=0.1,random_state = 42)
        elif(use_dataset=="mnist"):
            (train_images, train_labels),(test_images, test_labels) = mnist.load_data()
            train_images, val_images, train_labels, val_labels  = train_test_split(train_images,train_labels,test_size=0.1,random_state = 42)
      
        trainsize=train_images.shape[0]
        traind = train_images.reshape(trainsize,-1)
        mean = traind.mean(axis=0)
        normalized = (traind -  mean)/np.max((traind -  mean).max(axis=0))

        valsize=val_images.shape[0]
        val_images = val_images.reshape(valsize,-1)
        mean = val_images.mean(axis=0)
        val_images =  (val_images -  mean)/np.max((val_images -  mean).max(axis=0))

        testsize=test_images.shape[0]
        test_images = test_images.reshape(testsize,-1)
        max = (test_images -  mean).max(axis=0)
        test_images = (test_images -  mean)/np.max((test_images -  mean).max(axis=0))
        
        return normalized,test_images, val_images, train_labels,test_labels, val_labels