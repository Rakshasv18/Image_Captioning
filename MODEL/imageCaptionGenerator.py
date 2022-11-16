# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:22:24 2022

@author: Meeshawn Nithesh Raksha
"""

#%%  Import necessary packages
import numpy as np
import string
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
# from tensorflow.keras.layers import TextVectorization

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array

from tqdm import tqdm

seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)

#%% Defining Model Constants 

# Path to the images
IMAGES_PATH = '\dataset\Flickr8k_Dataset\Flicker8k_Dataset'
IMAGES_FOLDER = os.getcwd() + IMAGES_PATH

# Path to image captions
CAPTIONS_PATH = '\dataset\Flickr8k_text'
CAPTIONS_FOLDER = os.getcwd() + CAPTIONS_PATH

# Desired image dimensions
IMAGE_SIZE = (299, 299)

# Vocabulary size
VOCAB_SIZE = 10000

# Fixed length allowed for any sequence
SEQ_LENGTH = 25

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512

# Other training parameters
BATCH_SIZE = 64
EPOCHS = 30
AUTOTUNE = tf.data.experimental.AUTOTUNE

#%% Reading and storing the image filenames

def extractName(filename):
    file = open(filename, 'r')
    text = file.read()
    text = text.split('\n')
    file.close()
    return text    
    
#%%
train_imgs = extractName(CAPTIONS_FOLDER + 'Flickr_8k.trainImages.txt')
train_imgs = [x for x in train_imgs if x != '']

test_imgs = extractName(CAPTIONS_FOLDER + 'Flickr_8k.testImages.txt')
test_imgs = [x for x in test_imgs if x != '']

dev_imgs = extractName(CAPTIONS_FOLDER + 'Flickr_8k.devImages.txt')
dev_imgs = [x for x in dev_imgs if x != '']

#%% Loading images and extracting final layer features/weights

def extractFinalLayer(IMAGES_FOLDER, img_name, model):
    # Convert all the images to size 299x299 as expected by the
    # inception v3 model
    img = load_img(os.path.join(IMAGES_FOLDER, img_name), target_size=IMAGE_SIZE)
    # Convert PIL image to numpy array of 3-dimensions
    x = img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess images using preprocess_input() from inception module
    x = preprocess_input(x)
    x = model.predict(x)
    # reshape from (1, 2048) to (2048, )
    x = np.reshape(x, x.shape[1])
    return x


#%%
# Create an instance of the Inception V3 network
model_inceptionv3 = InceptionV3(weights='imagenet')
model_inceptionv3 = Model(model_inceptionv3.input, model_inceptionv3.layers[-2].output) 
finalLayer = extractFinalLayer(IMAGES_FOLDER,  train_imgs[0], model_inceptionv3)
print(finalLayer.shape)

#%%

dict_image_eigen_vector = {}
def featureExtractions(images):
    for image in tqdm(images):
        image_eigen_vectors = extractFinalLayer(IMAGES_FOLDER, image, model_inceptionv3)
        dict_image_eigen_vector[image] = image_eigen_vectors


#%%

featureExtractions(train_imgs)




#%% Read and store the image captions into a dictionary

file = open(CAPTIONS_PATH, 'r')
print('Reading and storing the image filenames and the corresponding captions\n' )

dict_descriptions = {}
for line in file:
    sentence = line.strip()
    sentence = sentence.split ('\t')   
    
    img_file_name = sentence[0].split('.')[0]
    caption = sentence[1]
    
    if dict_descriptions.get(img_file_name) == None:
        dict_descriptions[img_file_name] = []
     
    caption = 'BOS' + ' ' + caption + ' ' + 'EOS'
    dict_descriptions[img_file_name].append(caption)
    
file.close()
#%% Cleaning the dictionary:
maxLength = 0
for file, captions in dict_descriptions.items():
    for idx in range(len(captions)):       
        captions[idx] = captions[idx].lower()
        captions[idx] = captions[idx].translate(str.maketrans('', '', string.punctuation))
        captions[idx] = [word for word in captions[idx].split(' ') if len(word)>1]
        captions[idx] = [word for word in captions[idx] if word.isalpha()]
        captions[idx] = ' '.join(captions[idx])                
        if len(captions[idx].split(' ')) > maxLength:
                maxLength = len(captions[idx].split(' '))

#%% Create a dictionary of unique words:
vocabulary = {}
for key, captions in dict_descriptions.items():
    for caption in captions:
        for word in caption.split(' '):
            vocabulary[word] = vocabulary.get(word, 0) + 1
            
#%%
# word_count_thresh = 10
# reduced_vocabulary = []

# for word, count in vocabulary.items():
#     if count >= word_count_thresh:
#         print(word)
#         reduced_vocabulary.append(word)
        
#%% Writing the vocab list
with open('VocabList.txt', 'w') as f:
    for word in vocabulary:
        f.write(word)
        f.write('\n')
f.close()
        
#%% Creating Word embeddings for all the words in the vocabulary:
    
word_embeddings_matrix = {}
word_embeddings = {}

with open('\dataset\glove.6B') as f:
    for line in f:
        sentence = line.strip()
        sentence = sentence.split()
        word = sentence[0]
        feature_vector = sentence[1:]
        word_embeddings[word] = feature_vector

f.close()
        
for word in vocabulary:
    word_embeddings_matrix[word] = word_embeddings[word]
        



    
    
    
    