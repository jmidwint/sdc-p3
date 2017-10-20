
import numpy as np
import time

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


import pandas as pd
import matplotlib.pyplot as plt
import cv2
import code
import sys
import os
import csv
import math
from sklearn.model_selection import train_test_split

# Set Debug flag
JKM_DEBUG = True

# Set the random seed so as to be reproduceable during development
SEED = 5
DATADIR='data/' # data2ndtrackTake2/'
#RECORDED=True

def jkm_get_image(fn):
    # Get one image from disk 
    #    Images are assumed to be relative to the current working directory
    #    in data/
    DIR=DATADIR
    #if RECORDED: DIR=''   # I am given the full path if it is recorded by me.
    #if JKM_DEBUG: print("fn: ", fn)
    image = cv2.imread(DIR + fn, cv2.IMREAD_COLOR)  # Read the image as a color image. Default.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

#
# Some Helpers for Ananlyzing the Data
#
def jkm_display_image(img):
    # Display the image
    plt.ion()
    h = plt.imshow(img)
    h.axes.get_yaxis().set_visible(False)
    h.axes.get_xaxis().set_visible(False)
    # plt.gray()
    # plt.draw()
    # plt.show()
    #

def display_histogram(vals):
    plt.ion()
    histRange = plt.hist(vals, 50, normed=1)
    #plt.show()


# To Analyze the training results
def jkm_display_mse_graph(history):
    # Displays a grpah of the mean squared error for the training and validation 
    # results.
    # history is the object created during training
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

# Preprocessing Image Routines
orig_size_row, orig_size_col = 160, 320
nvidia_size_row, nvidia_size_col = 66, 200 # This is the nvidia model image size

# NOTE: This one also needs to be called from "drive.py" or equivalent, 
#        so that images we get from simulator camera are same when we pass to the
#        model.
def jkm_preprocess_image(image):
    # Preprocessing to crop image and resize it to be nvidia size
    #    instead of doing the cropping in the model. 
    #    input image np.array is orig size 160 x 320
    #    output image np.array is nvidia size 66 x 200
    #     * This should be called before running the model after training as well. From drive.py
    #     ** VERY IMPORTANT  
    shape = image.shape
    # The cropping below is moved from the keras model to here
    # model.add(Cropping2D(cropping=((32,25), (0,0)), input_shape=(160,320,3)))
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]] 
    image = cv2.resize(image,(nvidia_size_col,nvidia_size_row), interpolation=cv2.INTER_AREA)    
    return image

def jkm_flip_image(image, steer):
    # Randomly flip the image and the steering angle.
    #  This needs to be called after the image is already 
    #  preprocessed and has been cropped.
    #
    flip = np.random.randint(2)
    if flip==0:
        image = cv2.flip(image,1)
        steer = -steer
    return image, steer     

# Note these values come originally from the article here: 
#   https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
#IMAGE_HORIZ_SHIFT_PIXEL_RANGE = 100
STEERING_HORIZ_SHIFT_ADJUST = 0.4
DEBUG_SHIFT_IMAGE = True 
def jkm_shift_image (image, steer, pixel_range=100):
    # 1. Shift the image horizontally a random number of pixels within a pixel_range.
    #    Default pixel_range is 100, so shifting horizontally up to 50 pixels left or right. 
    # 2. Increase/Decrease the steering angle by a fractional value based on the 
    #    the random amount we are shifting the image, and the pixel_range and a constant 
    #    steering  horizontal adjustment factor. 
    # 3. Shift image vertically by a factor of a random amount up or down up by 20 pixels
    rows, cols, _ = image.shape   
    shift_x = pixel_range*np.random.uniform()-pixel_range/2
    steer_adj = shift_x/(pixel_range*STEERING_HORIZ_SHIFT_ADJUST )
    shift_steer = steer + (shift_x/(pixel_range*STEERING_HORIZ_SHIFT_ADJUST ))    
    print("really", (steer, steer_adj, shift_steer))
    shift_y = 40*np.random.uniform()-40/2
    #shift_y = 0
    shift_matrix = np.float32([[1,0,shift_x],[0,1,shift_y]])
    shift_image = cv2.warpAffine(image,shift_matrix,(cols,rows)) 
    if DEBUG_SHIFT_IMAGE:
        print("shift_x", shift_x)
        print("shift_steer",shift_steer)
        print("shift_y", shift_y)
        print("shift_matrix", shift_matrix)
    return shift_image, shift_steer


def jkm_preprocessing_data(data_line):
    ''' This run the image preprocessing for one line in the csv file.
        1. It randomly choses one of the 3 images & read it in, & adjusts the 
           steering angle accordingly.
        2. No other processing for now.
        3. Finally, do preprocessing on the image to crop it and re-size to fit
           into the model.  
    '''
    #
    # Get index into the data line. jkm - this is repetitive, speed this up later  
    # i_center, i_left, i_right, i_steer, _, _, _ = [i for i in range(len(data_col))]
    i_center, i_left, i_right, i_steer = 0, 1, 2, 3 
    # Get a random index into the images for left, right, centre
    #  We have to strip the white spaces because some of the names have spaces already 
    i_lrc = np.random.randint(3)
    if (i_lrc == i_center):
        fn = data_line[i_center].strip()
        shift_steer = 0.0
    if (i_lrc == i_left):
        fn = data_line[i_left].strip()
        shift_steer = 0.25
    if (i_lrc == i_right):
        fn = data_line[i_right].strip()
        shift_steer = -0.25
    # if JKM_DEBUG: print('Getting image %s and shift steer %f by %f' %(fn, data_line[i_steer], shift_steer))
    image = jkm_get_image(fn)
    steer = data_line[i_steer] + shift_steer 
    #
    # Crop & re-size image 
    image = jkm_preprocess_image(image)
    #
    # Randomly flip the image
    image, steer = jkm_flip_image(image, steer)  
    return image, steer
        
# Generator Definition for Preprocessing the images
#
from sklearn.utils import shuffle 
def jkm_generator(df_data, batch_size=32, hist=False):
    # Define the preprocessing to do on the image data.
    # this will be the generator to use during the keras model fit activity
    # to be able to generate the  data on the fly.
    #
    # df_data is of pandas df format
    #  Steps:
    #    1) Random Shuffle
    #    2) Get first 1000 samples 
    #    3) Randomly chose the left, middle , right 
    #    4) Make the adjustment to the steering depending if left, or right
    #
    len_df_data = len(df_data)
    #batch_images = np.zeros((batch_size, orig_size_row, orig_size_col, 3))
    #batch_steering = np.zeros(batch_size)
    while 1: # Loop forever so the generator never terminates
        df_data = shuffle(df_data)
        for offset in range(0, len_df_data, batch_size):
            # get batch size samples from randomly shuffled
            df_batch = df_data[offset:offset+batch_size]
            batch_images = []
            batch_steering = []
            #
            # Get each line in df_data, get one image & steer angle 
            # for i in range(batch_size):
            for data_line in df_batch.values:
                # data_line = df_batch.values[i]
                image, steering = jkm_preprocessing_data(data_line) # get 1 random image
                # later add additional augmentation calls here
                # move this call below to the jkm_preprocessing_data 
                # image = jkm_preprocess_image(image)  # crop and re-size to pass to model
                batch_images.append(image)
                batch_steering.append(steering)
                #batch_images[i] = image
                #batch_steering[i] = steering
            #    
            #
            # Return this one batch, but keeping note of where we
            #  are in the outer loop, so we continue from there 
            #  next time this generator runs. 
            X = np.array(batch_images)
            y = np.array(batch_steering)
            if hist:
                yield y
            else:   
                yield shuffle(X, y)

# My Model - based on nvidia model,
#   preprocessing within the model:
#     - crop top 1/5th (32 pixels) and bottom 20

# These numbers from article
CROP_TOP = 32  
CROP_BOTTOM = 25
input_shape = (160,320,3)

# This is the model definition
# Keras Imports
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Cropping2D, Lambda, Reshape, Convolution2D 
from keras.activations import relu
from keras.optimizers import Adam

#
# Run 1 Model
#  Use this to run it : python drive.feb28.nvidia.py model.h5
def jkm_nvidia_model():
    # Define the model.
    model = Sequential()
    # model.add(Cropping2D(cropping=((32,25), (0,0)), input_shape=(160,320,3)))
    #model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(103,320,3)))
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(nvidia_size_row, nvidia_size_col, 3))) 
    # 
    # 3 Convolution Layers, with 5x5 kernel and 2x2 strides
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid', init='he_normal' ))
    model.add(Activation('elu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid', init='he_normal' ))
    model.add(Activation('elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid', init='he_normal' ))
    model.add(Activation('elu'))
    # Then 2 Conv layers with 3x3 kernel and 1x1 strides
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', init='he_normal' ))
    model.add(Activation('elu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', init='he_normal' ))
    model.add(Activation('elu'))
    # Follows:  5 FC layers as per last 5 layers of nvidia model
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Activation('elu'))
    model.add(Dense(100))
    model.add(Activation('elu'))
    model.add(Dense(50))
    model.add(Activation('elu'))
    model.add(Dense(10))
    model.add(Activation('elu'))
    model.add(Dense(1))
    #
    # Nvidia Keras model doesn't have an activation at the end 
    #model.add(Activation('relu'))    
    #
    # Follows - commented out original simple model
    # model.add(Flatten())
    #model.add(Dense(128))
    #model.add(Activation('relu'))
    #model.add(Dense(1))
    #model.add(Activation('relu')) ** No
    #
    return model


''' Comment out for now 
def jkm_nvidia_model():
    # Define the model.
    model = Sequential()
    # Normalize
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(nvidia_size_row, nvidia_size_col, 3)))
    # 3 Convolution Layers, with 5x5 kernel and 2x2 strides
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid', init='he_normal' ))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid', init='he_normal' ))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid', init='he_normal' ))
    model.add(Activation('relu'))
    # Then 2 Conv layers with 3x3 kernel and 1x1 strides
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', init='he_normal' ))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid', init='he_normal' ))
    model.add(Activation('relu'))
    # Follows:  5 FC layers
    model.add(Flatten())
    model.add(Dense(1164, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(100, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(50, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(10, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dense(1, init='he_normal'))
    model.add(Activation('relu'))    
    #
    return model
'''

# Print out the model definition and layer shapes	
def jkm_check_model(model):
    num_layers = len(model.layers)
    assert num_layers != 0, 'No layers found'
    for i in range(num_layers):
        in_shape, out_shape = model.layers[i].input_shape, model.layers[i].output_shape
        layer_name = model.layers[i].get_config()['name']        
        print("Layer: %d %s in: %s out: %s" %(i,layer_name,  in_shape, out_shape ,)) 


# MAIN
# ====
def main(argv=None):  # pylint: disable=unused-argument

    # Run everything inside of main
    np.random.seed(SEED)
    print("Setting seed to %d" %SEED)


    # Step 1: Load the data 
    # ========================
    # 
    print("Reading csv file")
    df = pd.read_csv(DATADIR +'driving_log.csv')
    df_train_samples, df_validation_samples = train_test_split(df, random_state = SEED, test_size=0.2)
    print("Train and validation df split completed")
    '''
    # Set index into the values
    idx_center, idx_left, idx_right, idx_steering, idx_throttle, idx_brake, idx_speed = [i for i in range(len(df.columns))]
    # Get the Data Values
    # Just going to use the centre images for now
    y_train = df.values[:, idx_steering]
    X_train_img = df.values[:, idx_center] # just get the file names for now
    '''
    
    print("\n Step_1__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")
    
    # Step 2: Define the model
    # =========================
    print("Defining Model")
    model = jkm_nvidia_model()
    print("Model Defined")
    
    if JKM_DEBUG: jkm_check_model(model)
    
    
    # Step 3: Compile Model
    # ======================
    print("Compiling Model")
    #adam = model.optimizer.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adam = Adam(lr=0.0001) 
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
    
    
    print("\n Step_3__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")
    
    
    # Step 4: Fit/Train Model
    # =======================
    # train the model using the generator function
    
    print("Create training and validation generators")
    train_generator = jkm_generator(df_train_samples, batch_size=32)
    validation_generator = jkm_generator(df_validation_samples, batch_size=32)
    
    NUM_EPOCHS = 20  # was 20 was 10 was 5  - try 40 for 2nd track, was 20  
    print("Starting Training for epochs: ", NUM_EPOCHS)
    
    history = model.fit_generator(train_generator, 
                                  samples_per_epoch=len(df_train_samples), 
                                  validation_data=validation_generator,
                                  nb_val_samples=len(df_validation_samples), verbose=1,
                                  nb_epoch=NUM_EPOCHS)

        
    
    train_acc = history.history['acc'][-1]
    val_acc = history.history['val_acc'][-1]
    print("\n Accuracy: %4f" %train_acc)
    print("Val Accuracy: %4f" %val_acc)
    print("Training Complete")

    # display training mse results in a graph
    jkm_display_mse_graph(history)

   
    print("\n Step_4__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")
    
    
    
    # Step 5: Evaluate Model:
    # =======================
    # JKM - this is just testing on same data as training
    # JKM - fix this 
    # print("Running evaluation on test data")
    # score = model.evaluate(X_test_normalized, y_test_one_hot)
    #score = model.evaluate(X, y)drive.simple.py
    #print('Test score:', score[0])
    #print('Test accuracy:', score[1])
    
    
    print("\n Step_5__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  
    
    
    # Step 6: Capture the model
    # ==========================
    
    print("Saving model to model.h5")
    model.save('model.h5')
    
    # with open('model.json', 'w') as f:
    #    f.write(model.to_json())
    # model.save_weights('model.h5')


    print("\n Step_6__Program paused. Press Ctrl-D to continue.\n")
    code.interact(local=dict(globals(), **locals()))
    print(" ... continuing\n ")  
    

# To run this as a script. 
if __name__ == '__main__':
    tf.app.run()


'''
# 
# ==============================================================================
#
# Archive Procs
# =============

def jkm_proto_get_3_test_data():
    # get 3 images & angles for prototyping model & testing it
    # I manually looked at csv file and grabbed data from 3 
    fn_1 = 'IMG/center_2016_12_01_13_32_44_569.jpg' # 64 centre_go_straight
    fn_2 = 'IMG/center_2016_12_01_13_32_43_862.jpg' # 57 centre_go_right
    fn_3 = 'IMG/center_2016_12_01_13_32_52_652.jpg' # 144 centre_go_left
    img_1 = jkm_get_image(fn_1)
    img_2 = jkm_get_image(fn_2)
    img_3 = jkm_get_image(fn_3)
    X =  np.asarray([img_1, img_2, img_3])
    jkm_display_image(img_1)
    jkm_display_image(img_2)
    jkm_display_image(img_3) 
    steer_1 = 0.9855326
    steer_2 = 0.5784606
    steer_3 = -0.1547008
    y = np.asarray([steer_1, steer_2, steer_3])
    return X, y 

# JKM - not sure if I want to get in the data this way
# JKM - unused, remove later
from sklearn.model_selection import train_test_split
def jkm_get_csv_and_split():
    # Read the csv data and split it into training and validation data
    #  return as np.arrays of csv lines 
    samples = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    train_samples, validation_samples = train_test_split(samples, random_state = SEED, test_size=0.2)
    return np.array(train_samples), np.array(validation_samples)

# Archive Models
# ===============

# 

# This one to drive around the track completely
#   python drive.feb28.nvidia.py model.h5
#   cp model.h5 model.aroundtrack.h5
#   cp model.py model.aroundtrack.py
def jkm_nvidia_model():
    # Define the model.
    model = Sequential()
    # model.add(Cropping2D(cropping=((32,25), (0,0)), input_shape=(160,320,3)))
    #model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(103,320,3)))
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(nvidia_size_row, nvidia_size_col, 3))) 
    # 
    # Follows:  5 FC layers as per last 5 layers of nvidia model
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Activation('elu'))
    model.add(Dense(100))
    model.add(Activation('elu'))
    model.add(Dense(50))
    model.add(Activation('elu'))
    model.add(Dense(10))
    model.add(Activation('elu'))
    model.add(Dense(1))
    #
    # Nvidia Keras model doesn't have an activation at the end 
    #model.add(Activation('relu'))    
    #
    # Follows - commented out original simple model
    # model.add(Flatten())
    #model.add(Dense(128))
    #model.add(Activation('relu'))
    #model.add(Dense(1))
    #model.add(Activation('relu')) ** No
    #
    return model




#  Use this to run it : python drive.feb28.nvidia.py model.h5
#  This model does preprocessing of image outside of the model inside the generator
def jkm_nvidia_model():
    # Define the model.
    model = Sequential()
    # model.add(Cropping2D(cropping=((32,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(103,320,3)))
    #model.add(Reshape((66, 200, 3)))
    # Simple prototype here for now
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    #
    return model



#  Use this to run it : drive.simple.pypython drive.simple.py model.h5.simple
def jkm_nvidia_model():
    # Define the model.
    model = Sequential()
    model.add(Cropping2D(cropping=((32,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 127.5) - 1.0))
    #model.add(Reshape((66, 200, 3)))
    # Simple prototype here for now
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    #
    return model


# This was the first one and very simple and worled to get 
#  my car over teh bridge ok. 
def jkm_nvidia_model_1():
    # Define the model.
    model = Sequential()
    model.add(Cropping2D(cropping=((32,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 127.5) - 1.0))
    #model.add(Reshape((66, 200, 3)))
    # Simple prototype here for now
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    #
    return model

# never got this one working 
#  due to issue with resizing an imnage from within a keras model
def jkm_nvidia_model_2():
    # Define the model.
    model = Sequential()
    model.add(Cropping2D(cropping=((32,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(jkm_resize_image))
    model.add(Lambda(tf.image.resize_images,
                 output_shape=(nvidia_size_row, nvidia_size_col),
                 arguments={'size': (nvidia_size_row, nvidia_size_col)}))
    model.add(Lambda(lambda x: (x / 127.5) - 1.0))
    #model.add(Reshape((66, 200, 3)))
    # Simple prototype here for now
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    #
    return model





# Archive Code
#
# try to use this as a Lamda function
# from keras import backend as K
# This was from trying to get my model to do the re-sizing inside the keras model.
#  but bug with keras makes this non functional.
def jkm_resize_image(x):
    # Take an image and re-size it to the nvidia size image
    #import tensorflow as tf
    # tf.python.control_flow_ops = tf
    image = tf.image.resize_images(x, nvidia_size_row, nvidia_size_col )
    print("image new shape:" , image.get_shape())
    return image


    #if JKM_DEBUG: print("type: ", type(x))
    #print ("Shape :" , x.get_shape())
    #image = x[0, :, :, :]
    #print("x is keras tensor" , K.is_keras_tensor(x))
    #print("image is keras tensor: ", K.is_keras_tensor(image))
    #print ("image type", type(image))
    #print ("image shape", image.get_shape)
    #ses = tf.InteractiveSession()
    #array = image.eval(session=ses)
    #ses.graph.get_operation_by_name('operation')
    # print( ses.graph.get_operations())
    #graph = tf.get_default_graph()
    # operations = graph.get_operations()
    # print("operations: ", operations)
    #if JKM_DEBUG: print(type(new_image))
    #print (new_image) 
    #x = cv2.resize(x,(nvidia_size_col,nvidia_size_row), interpolation=cv2.INTER_AREA)    
    #return x

from itertools import islice
list(islice(histogram_generator, 1, 2))

#list(islice(infinite_counter(), 1000, 2000))

for i in range(5):
      image_list, steer_list = next(histogram_generator)

print(len(steer_list))
print(steer_list)  

JKM under construction
def jkm_data_for_histogram(df_data, num_samples):
    # Gather random data to create a histogram
    # df_data is of pandas df format
    #  Steps:
    #    1) Random Shuffle
    #    2) Get first 1000 samples 
    #    3) Randomly chose the left, middle , right 
    #    4) Make the adjustment to the steering depending if left, or right
    len_df_data = len(df_data)
    df_data = shuffle(df_data)
    df_batch = df_data[0:num_samples]
    batch_images = []
    batch_steering = []
    #
    # Get each line in df_data
    # for i in range(batch_size):
            for data_line in df_batch.values:
                # data_line = df_batch.values[i]
                image, steering = jkm_preprocessing_data(data_line) # get random image
                # later add additional augmentation calls here
                image = jkm_preprocess_image(image)  # crop and re-size to pass to model
                batch_images.append(image)
                batch_steering.append(steering)
                #batch_images[i] = image
                #batch_steering[i] = steering
                
            #
            # Return this one batch, but keeping note of where we
            #  are in the outer loop, so we continue from there 
            #  next time this generator runs. 
            X = np.array(batch_images)
            y = np.array(batch_steering)
            yield shuffle(X, y)


'''
