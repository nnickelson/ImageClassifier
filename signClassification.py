# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 03:09:43 2019

@author: Nathan
"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import copy
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 16, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('signTrainingSet',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('signTestSet',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

max_predict = 0
for n in range(10,20,1):
    newClassifier = copy.deepcopy(classifier)
    print(len(training_set), "*****")
    newClassifier.fit_generator(training_set,
                             steps_per_epoch = 16,
                             epochs = n,
                             validation_data = test_set,
                             validation_steps = 16)
    # Part 3 - Making new predictions
    import numpy as np
    from keras.preprocessing import image
    corr = 0
    for i in range(31):
        if i%2 == 0:
            actual = 'yield'
        else:
            actual = 'stop'    
        test_image = image.load_img('stopyield\stopyield'+str(i)+'.jpg', target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = newClassifier.predict(test_image)
        training_set.class_indices
        if result[0][0] == 1:
            prediction = 'yield'
        else:
            prediction = 'stop'
        if (prediction==actual):
            corr = corr + 1
        #print(prediction, prediction==actual)
    print("Correct: ",corr, 31, corr/31)
    if corr/30 > max_predict:
        max_predict = corr/31
        best_classifier = copy.deepcopy(newClassifier)

    
    
    
    
    
    

