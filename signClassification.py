# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 03:09:43 2019

@author: Nathan
"""

# Importing the Keras libraries and packages
import sys
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import copy
import numpy as np
import matplotlib.pyplot as plt


# inputs I have changed to test machines with different parameters
filters = 16
pixelSize = 128
hidden = 64


# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(filters, (3, 3), input_shape = (pixelSize, pixelSize, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(filters, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = hidden, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('signTrainingSet',
                                                 target_size = (pixelSize, pixelSize),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('signTestSet',
                                            target_size = (pixelSize, pixelSize),
                                            batch_size = 32,
                                            class_mode = 'binary')

max_predict = 0
epchs = 0

## Originally setup to run loops to find the best machine with different epochs
## Currently just set to run a 25 epoch machine and copy it
for n in range(15,16,1):
    newClassifier = copy.deepcopy(classifier)
    print(len(training_set), "*****")
    newClassifier.fit_generator(training_set,
                             steps_per_epoch = 32,
                             epochs = n,
                             validation_data = test_set,
                             validation_steps = 32)
    # Part 3 - Making new predictions
    print("LETS GET STARTED!!!!")
    corr = 0
    for i in range(60):
        if i%2 == 0:
            actual = 'yield'
        else:
            actual = 'stop'    
        test_image = image.load_img('stopyield\stopyield'+str(i)+'.jpg', target_size = (pixelSize, pixelSize))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = newClassifier.predict(test_image)
        print(result[0][0])
        training_set.class_indices
        if result[0][0] == 1:
            prediction = 'yield'
        else:
            prediction = 'stop'
        if (prediction==actual):
            corr = corr + 1
        #print(prediction, prediction==actual)
    print("Correct: ",corr, 60, corr/60)
    if corr/60 > max_predict and n > epchs:
        max_predict = corr/61
        epchs = n
        best_classifier = copy.deepcopy(newClassifier)


# This section list the results from shading a picture
# If the shading results cause a classifier to flip an image, its results will be listed

print("")
for t in range(60):
    print("should be " + str((t+1)%2) )
    print("****Picture " + str(t) + "*****")
    for s in range(25):
        #print("shading = {}".format(100-2*s/100))
        
        test1 = image.load_img('stopyield\stopyield' + str(t) + '.jpg', target_size = (pixelSize, pixelSize))
        test1 = image.img_to_array(test1)
        test1 = np.expand_dims(test1, axis = 0)
        #training_set.class_indices
        result1 = best_classifier.predict(test1)
        
        shade_image = image.load_img('stopyield\stopyield' + str(t) + '.jpg', target_size = (pixelSize, pixelSize))
        
        shade_image = shade_image.point(lambda p: p*(100-4*s)/100)
        shade_image = image.img_to_array(shade_image)
        shade_image = np.expand_dims(shade_image, axis = 0)
        #training_set.class_indices
        result2 = best_classifier.predict(shade_image)
        
        if result1 != result2:
            print("shading = {} .... r1 = {} .... r2 = {}".format((100-4*s)/100, result1[0][0], result2[0][0]))
    print("Press 'enter' for next results, 'n' to skip to red/bg shaders, or 'q' to quit")    
    stp = input()
    if stp == "q":
        sys.exit()
    if stp == "n":
        break

# This section combines the red shading results with the blue green shading results
# For each image, the shade scale is listed followed by the prediction of the unedited image
# afterward the predictions after red-shading and blue-green shading are shown at each level
# of shading

for t in range(60):
    sygn = ""
    if (t+1)%2 == 0:
        sygn = "stop"
    else:
        sygn = "yield"
    print("should be " + str((t+1)%2) )
    print("****Picture " + str(t) + "***** " + sygn + " *****" )
    for s in range(20):
        #print("shading = {}".format(100-2*s/100))
        
        test1 = image.load_img('stopyield\stopyield' + str(t) + '.jpg', target_size = (pixelSize, pixelSize))
        test1 = image.img_to_array(test1)
        test1 = np.expand_dims(test1, axis = 0)
        #training_set.class_indices
        result1 = best_classifier.predict(test1)
        
        test2 = image.load_img('stopyield\stopyield' + str(t) + '.jpg', target_size = (pixelSize, pixelSize))
        test3 = image.load_img('stopyield\stopyield' + str(t) + '.jpg', target_size = (pixelSize, pixelSize))
        
        rScale = s/4
        for x in range(pixelSize):
            for y in range(pixelSize):
                p2 = test2.getpixel((x,y,))
                p3 = test3.getpixel((x,y,))
                
                # red tints test2 image
                if p2[0] * rScale > 255:
                    test2.putpixel( (x,y), (255, p2[1], p2[2]))
                else:
                    test2.putpixel( (x,y), (int(p2[0]*rScale), p2[1],p2[2]))
                
                # blue-green tints teste image 
                test3.putpixel( (x,y), (p3[0], min([int(p3[1]*rScale), 255]), min([int(p3[2]*rScale), 255])  ) )
        if s == 8:
            test2.show()
            test3.show()
            
        test2 = image.img_to_array(test2)
        test2 = np.expand_dims(test2, axis = 0)
        
        test3 = image.img_to_array(test3)
        test3 = np.expand_dims(test3, axis = 0)
        
        #training_set.class_indices
        result2 = best_classifier.predict(test2)
        result3 = best_classifier.predict(test3)
        
        print("shadeScale = {} .. or.Pred. = {} -- redImg = {} -- bgImg = {}".format(rScale, result1[0][0], result2[0][0], result3[0][0]))
    print("Press 'enter' for next results or 'q' to quit")    
    stp = input()
    if stp == "q":
        sys.exit()

'''

#example code for red shading a pixel  

for x in range(64):
    for y in range(64):
        p = test3.getpixel((x,y,))
        if p[0] * 5 > 255:
            test3.putpixel( (x,y), (255, p[1], p[2]))
        else:
            test3.putpixel( (x,y), (p[0]*5, p[1],p[2]))
'''
#Example to darken image

#load image - target_size parameter optional
#bob = image.load_img('stopyield\stopyield1.jpg', target_size=(64,64)) 

# reduce intensity of pixels by 50% in this example
#bob2 = bob.point(lambda p: p*0.5) 

# show the darkened image
#bob2.show()
#print("Should be zer0")


























