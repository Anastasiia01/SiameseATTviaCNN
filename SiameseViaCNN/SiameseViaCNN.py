import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from Siamese import Siamese
import numpy as np
import os
import cv2
#from tensorflow.keras.utils import to_categorical

def visualize(embed, labels):
    labelset = set(labels)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    for label in labelset:
        indices = np.where(labels == label)
        ax.scatter(embed[indices,0], embed[indices,1], label = label, s = 20)
    ax.legend()
    #fig.savefig('embed.jpeg', format='jpeg', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    # Load AT&T dataset
    train = np.empty((200,112,92),dtype='float64')
    trainY = np.zeros((200,))
    test = np.empty((200,112,92),dtype='float64')
    testY = np.zeros((200,))

    # Load in the images
    i = 0
    for filename in os.listdir('C:/Users/anast/Documents/Computer-Vision/AttDataSet/ATTDataSet/Training/'):
        y = int(filename[1:filename.find('_')])-1
        trainY[i]=y
        #trainY[i,y] = 1.0 one hot binary
        train[i] = cv2.imread('C:/Users/anast/Documents/Computer-Vision/AttDataSet/ATTDataSet/Training/{0}'.format(filename),0)/255.0 # 0 flag stands for greyscale; for color, use 1
        i = i + 1
    i = 0 # read test data
    for filename in os.listdir('C:/Users/anast/Documents/Computer-Vision/AttDataSet/ATTDataSet/Testing/'):
        y = int(filename[1:filename.find('_')])-1
        testY[i]=y
        #testY[i,y] = 1.0 one hot binary
        test[i] = cv2.imread('C:/Users/anast/Documents/Computer-Vision/AttDataSet/ATTDataSet/Testing/{0}'.format(filename),0)/255.0
        i = i + 1
    print(trainY.shape)
    #cv2.imshow('image',train[0])
    #cv2.waitKey(0)
    trainX = train.reshape(train.shape[0],train.shape[1],train.shape[2],1)
    testX = test.reshape(test.shape[0],test.shape[1],test.shape[2],1)
    #resized=cv2.resize(train[0], (92,92), interpolation=cv2.INTER_LINEAR)
    siamese = Siamese()
    siamese.trainSiamese(trainX, trainY, 10,20)
    siamese.trainSiameseForClassification(trainX, trainY, 20,20)
    
    # Test model
    embed = siamese.test_model(input = testX)
    siamese.computeAccuracy(testX,testY)

if __name__ == '__main__':
    main()


