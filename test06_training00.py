#!/usr/bin/env python
# coding: utf-8

# In[18]:


#MAIN TEST FILE
#reading whole images


import cv2 as cv2   #importing opevcv
import numpy as np
import csv
import os

path = "dataset"
ext = [".jpeg", ".jpg", ".png"]
numOfContours=0

#getting all filenames in a folder
fileNameExtList = []
fileNameOnlyList = []
for file in os.listdir(path):
    if file.endswith(tuple(ext)):
        fileNameExtList.append(file)
        #print(os.path.join(path, file)) #== 'symptoms/11085_left.jpeg'
        fileName, extension = os.path.splitext(file)
        fileNameOnlyList.append(fileName)

#ROI manual specs
x = 65
y = 65
w = 300
h = 190

#writing to a csv file
csvTitle = [['image_name', 'count', 'area']]
csvData = []

contourThreshold = 5

for fileName in fileNameExtList:
    img = cv2.imread(os.path.join(path,fileName))    #reading an image
    img_rs= cv2.resize(img, (430, 320))     #orginal image resizing
    #cv2.imshow('orginal', img_rs)
    print(fileName)
    
    imCrop = img_rs[y:y+h, x:x+w]
    #cv2.imshow("Image", imCrop)
    
    hsv_im = cv2.cvtColor(imCrop, cv2.COLOR_BGR2HSV)  #converting the  image to HSV
    clr_rng_img = cv2.inRange(hsv_im,(19,0,250), (255,255,255)) 
    #cv2.imshow('color range orginal', clr_rng_img);    
    
    #contour drawing test code
    image, contours, hierarchy = cv2.findContours(clr_rng_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    numOfContours1 = len(contours)   #number of contours
    
    if numOfContours1 > contourThreshold:
        #19,0, 111
        clr_rng_img = cv2.inRange(hsv_im,(19,0,111), (255,255,255)) 
    
        #contour drawing test code
        image, contours, hierarchy = cv2.findContours(clr_rng_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        numOfContours = len(contours)   #number of contours
    
    
        print("contours = ",numOfContours)
        area = []

        for contour in contours:
            x1,y1,w1,h1 = cv2.boundingRect(contour)
            cv2.rectangle(imCrop, (x1,y1), (x1+w1, y1+h1), (0,0,255), 1)
            area.append(w1*h1)

        try:
            maxArea = max(area)
            print("max area = ", maxArea)
        except:
            maxArea = 0
        try:
            area.remove(maxArea)
        except:
            maxArea = 0
        sumOfArea = sum(area)
        print("sum of Area =",sumOfArea)
        #cv2.imshow("Bounding box",imCrop)

        csvData.append([fileName, numOfContours, sumOfArea])
   
    else:

        print("contours = ",numOfContours)
        area = []

        for contour in contours:
            x1,y1,w1,h1 = cv2.boundingRect(contour)
            cv2.rectangle(imCrop, (x1,y1), (x1+w1, y1+h1), (0,0,255), 1)
            area.append(w1*h1)

        
        try:
            maxArea = max(area)
            print("max area = ", maxArea)
        except:
            maxArea = 0
        try:
            area.remove(maxArea)
        except:
            maxArea = 0
        sumOfArea = sum(area)
        print("sum of Area =",sumOfArea)

        csvData.append([fileName, numOfContours, sumOfArea])
        
        
with open('dr_features.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvTitle)
    writer.writerows(csvData)


#keyboard binding function
k = cv2.waitKey(0);   
if k == 27:           #wait for ESC key to exit
    cv2.destroyAllWindows();     #destroys all windows created
    
    
    
    
    
    


# In[42]:


#TEST 01
#trying to write to csv file
#training the above code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


data=pd.read_csv("dr_features.csv")
data_new=pd.read_csv("dr_features.csv")

predictions=data_new['count']
#data_new

threshold = 30
data_new['pred_value'] = predictions.apply(lambda x: 1 if x > threshold else 0)



features_raw = data_new[[ "count", "area"]]
from sklearn.model_selection import train_test_split

predict_class = data_new['pred_value']



np.random.seed(100)

X_train, X_test, y_train, y_test = train_test_split(features_raw, predict_class, train_size=0.80, random_state=1)


# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


import sklearn
from sklearn import svm

C = 1.0
# Create SVM classification object 
#svc = svm.SVC(kernel='linear',C=C,gamma=2)
svc = svm.SVC(kernel='linear',C=C,gamma=2)

svc.fit(X_train, y_train)


from sklearn.metrics import fbeta_score
predictions_test = svc.predict(X_test)
print(predictions_test)


from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions_test))
print(classification_report(y_test, predictions_test))



plt.subplot(221), plt.hist(predict_class, bins=10)
plt.xlim(0,1)
plt.title('Histogram of disease prediction')
plt.xlabel('prediction class')
plt.ylabel('Frequency')

plt.subplot(222), plt.hist(predictions_test, bins=10)
plt.xlim(0,1)
plt.xlabel('prediction test')
plt.ylabel('Frequency')

data_new.plot(data_new["count"], data_new["area"],kind="scatter",color="red")
plt.subplot(223), plt.show()

count=0
with open('dr_features.csv','r') as csvinput:
    with open('dr_features_output.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('pred_values')
        all.append(row)

        for row in reader:
            row.append(predict_class[count])
            count = count + 1
                           
            all.append(row)

        writer.writerows(all)
        
        



# In[49]:


#TEST 02
#Same above code
#for test a single image value


import cv2 as cv2   #importing opevcv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


predictions=data_new['count']
#data_new

numOfContours=0


#start test code

#imgPath = input("enter path/name.ext of the image")

img = cv2.imread(os.path.join("dataset/11085_left.jpeg"))    #reading an image
img_rs= cv2.resize(img, (430, 320))     #orginal image resizing
cv2.imshow('orginal', img_rs)
    
#ROI manual specs
x = 65
y = 65
w = 300
h = 190

imCrop = img_rs[y:y+h, x:x+w]
cv2.imshow("Image", imCrop)
hsv_im = cv2.cvtColor(imCrop, cv2.COLOR_BGR2HSV)  #converting the  image to HSV
    
clr_rng_img = cv2.inRange(hsv_im,(19,0,111), (255,255,255)) 
    
#contour drawing test code
image, contours, hierarchy = cv2.findContours(clr_rng_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
numOfContours = len(contours)   #number of contours
    
#print("contours = ",numOfContours)
area = []

for contour in contours:
    x1,y1,w1,h1 = cv2.boundingRect(contour)
    cv2.rectangle(imCrop, (x1,y1), (x1+w1, y1+h1), (0,0,255), 1)
    area.append(w1*h1)

    try:
        maxArea = max(area)
        #print("max area = ", maxArea)
    except:
        maxArea = 0
    try:
        area.remove(maxArea)
    except:
        maxArea = 0
            
    #sumOfArea = sum(area)
    #print("sum of Area =",sumOfArea)
    #cv2.imshow("Bounding box",imCrop)

print(area)
print(sum(area))
print(numOfContours)
input_data = pd.DataFrame([[100,50]])

print(input_data)
#End test cod




predictions_test = svc.predict(input_data)
print(predictions_test)

if(predictions_test):
    print("The patient is DIABETIC")
else:
    print("NOT DIABETIC")


        


# In[ ]:




