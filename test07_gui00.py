#!/usr/bin/env python
# coding: utf-8

# In[2]:


#TEST 01
#trying to write to csv file
#training the above code

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import csv

import cv2 as cv2   #importing opevcv
import os
import numpy as np



#training the model with above generated csv file

#import test061_training00

data=pd.read_csv("dr_features_output_main.csv")
data_new=pd.read_csv("dr_features_output_main.csv")

predictions=data_new['count']
#data_new

threshold = 30
#data_new['pred_value'] = predictions.apply(lambda x: 1 if x > threshold else 0)
#data_new['pred_value00'] = data_new['pred_value']


features_raw = data_new[[ "count", "area"]]
predict_class = data_new['pred_values']


from sklearn.model_selection import train_test_split


np.random.seed(100)

X_train, X_test, y_train, y_test = train_test_split(features_raw, predict_class, train_size=0.80, random_state=1)

# Show the results of the split
#print "Training set has {} samples.".format(X_train.shape[0])
#print "Testing set has {} samples.".format(X_test.shape[0])


import sklearn
from sklearn import svm

C = 5
# Create SVM classification object 
svc = svm.SVC(kernel='linear',C=C,gamma=2)

svc.fit(X_train, y_train)


from sklearn.metrics import fbeta_score
predictions_test = svc.predict(X_test)
print(predictions_test)


from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions_test))
print(classification_report(y_test, predictions_test))



def reading(imageSource):
    
    numOfContours=0
    imgPath = imageSource
    #raw_input("enter path/name.ext of the image")

    img = cv2.imread(os.path.join(imgPath))    #reading an image
    img_rs= cv2.resize(img, (430, 320))     #orginal image resizing
    #cv2.imshow('orginal', img_rs)

    #ROI manual specs
    x = 65
    y = 65
    w = 300
    h = 190

    imCrop = img_rs[y:y+h, x:x+w]
    #cv2.imshow("Image", imCrop)
    hsv_im = cv2.cvtColor(imCrop, cv2.COLOR_BGR2HSV)  #converting the  image to HSV

    clr_rng_img = cv2.inRange(hsv_im,(19,0,111), (255,255,255)) 
    #cv2.imshow("Binarise", clr_rng_img)

    #contour drawing test code
    image, contours, hierarchy = cv2.findContours(clr_rng_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    numOfContours = len(contours)   #number of contours

    #print("contours = ",numOfContours)
    area1 = []
    sumOfArea = 0 
    for contour in contours:
        x1,y1,w1,h1 = cv2.boundingRect(contour)
        cv2.rectangle(imCrop, (x1,y1), (x1+w1, y1+h1), (0,0,255), 1)
        area1.append(w1*h1)

        try:
            maxArea = max(area)
            #print("max area = ", maxArea)
        except:
            maxArea = 0
        try:
            area11.remove(maxArea)
        except:
            maxArea = 0

        sumOfArea = sum(area1)
        #print("sum of Area =",sumOfArea)
        #cv2.imshow("Bounding box",imCrop)

    print(sumOfArea)
    print(numOfContours)
    input_data = pd.DataFrame([[numOfContours, sumOfArea]])

    print(input_data)
    #End test cod

    predictions_test1 = svc.predict(input_data)
    print(predictions_test1)

    if(predictions_test1):
        print("The patient is DIABETIC")
    else:
        print("NOT DIABETIC")




    #keyboard binding function
    k = cv2.waitKey(0);   
    if k == 27:           #wait for ESC key to exit
        cv2.destroyAllWindows();     #destroys all windows created


    

