#!/usr/bin/env python
# coding: utf-8

# # 1. Installing Dependencies and Setup

# In[2]:


from IPython.display import Image, display

# Specify the path to the image file or use the filename if it's in the same directory.
image_path = 'D:\Waste-classifier/cpu config tflow.png'

# Display the image in the Jupyter Notebook
display(Image(filename=image_path))


# In[3]:


pip install tensorflow==2.12.0


# In[4]:


pip install keras==2.10.0


# In[5]:


from IPython.display import Image, display

# Specify the path to the image file or use the filename if it's in the same directory.
image_path = 'D:\Waste-classifier/gpu config tflow.png'

# Display the image in the Jupyter Notebook
display(Image(filename=image_path))


# In[6]:


pip install tensorflow-gpu==2.10.0


# In[7]:


pip install opencv-python


# In[8]:


pip install matplotlib


# In[9]:


get_ipython().system('pip list')


# In[10]:


import tensorflow as tf
import os


# In[11]:


# to see all gpu available in the system
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
print(len(gpus))     # to get number of gpu available


# In[12]:


# to see all cpu available in the system
gpus = tf.config.experimental.list_physical_devices('CPU')
print(gpus)
print(len(gpus))     # to get number of cpu available


# In[13]:


import cv2
import imghdr


# In[14]:


data_dir = 'train'   #gives path to directory


# In[15]:


os.listdir('train')   #list folders inside


# In[16]:


# to see number of files harzardous folder
print(len(os.listdir(os.path.join('train','r'))))


# In[17]:


# to see number of files recyclable folder
print(len(os.listdir(os.path.join('train','nr'))))


# In[18]:


# immage extensions that work
image_exts = ['jpeg','jpg','bmp','png']


# In[19]:


from matplotlib import pyplot as plt


# In[20]:


# reding an image using cv2
img = cv2.imread(os.path.join('train','r', 'R_1.jpg'))
print(img)
# reads images numpy array


# In[21]:


plt.imshow(img)
# image would be bluish as opencv reads image as bgr not rgb


# In[22]:


# to color it
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# # 2. Performing Cleanup of bad images

# In[23]:


# looping through each image to remove bad images
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image does not exist in extension list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print("Exception occured, ",e)
            print('Issue with image {}'.format(image_path))


# In[24]:


# after cleanup number of images left
# to see number of files harzardous folder
print(len(os.listdir(os.path.join('train','r'))))


# In[25]:


# to see number of files recyclable folder
print(len(os.listdir(os.path.join('train','nr'))))


# # 3. Building a Data pipeline
# Processing data in 2 classes namely 'Recyclable' , 'Non-Recyclable'

# In[26]:


import numpy as np
from matplotlib import pyplot as plt


# In[27]:


# data has 2 classes or is classified into 2 distinct classes


# In[28]:


# builds a data pipeline
data = tf.keras.utils.image_dataset_from_directory('train')


# In[29]:


# allows us to access the pipeline
data_iterator = data.as_numpy_iterator()


# In[30]:


# accessing the pipeline
# creates a group  or collection of images
batch = data_iterator.next()


# In[31]:


# a batch(group of data)
# a set of images made to perform process effeciently
batch


# In[32]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# In[33]:


# the above result shows
# 0 - Non - Recyclable
# 1 - Recyclable


# # 3. Preprocessing Data
# Cleaning, transforming, and preparing raw data to make it suitable to be used as input for deep learning model

# In[34]:


scaled = batch[0] / 255


# In[35]:


scaled


# In[36]:


scaled.min()


# In[37]:


scaled.max()


# In[38]:


# scaling data
data = data.map(lambda x,y: (x/255, y))


# In[39]:


scaled_iterator = data.as_numpy_iterator()
# does shuffling


# In[40]:


batch = scaled_iterator.next()


# In[41]:


batch[0].max()


# In[42]:


batch[0].min()


# In[43]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])


# In[44]:


len(data)


# In[45]:


train_size = int(len(data)*.7)   # training set
val_size = int(len(data)*.2) + 1  # validation set
test_size = int(len(data)*.1) + 1  # testing set


# In[46]:


train_size


# In[47]:


val_size


# In[48]:


test_size


# In[49]:


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size).take(test_size)


# # 4. Building Deep Learning Model
# Using neural networks with multiple layers to learn patterns and make predictions

# In[50]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[51]:


model = Sequential()


# In[52]:


# other way
# more readable and clean
# 16- filters
# 3x3 size filters
# 1 - stride

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[53]:


# adam - optimizer
# defining losses
# binary classification
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


# In[54]:


model.summary()


# # 5. Training the Model
# Iteratively optimizing model parameters using labeled data to minimize error and enable accurate predictions on new data

# In[55]:


logdir = 'logs'


# In[56]:


# callback important if we want to save the model at a particular checkpoint
# seeing the model
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[57]:


# 2 important methods of building a neural network
# model.fit - fit is a training component
# model.predict - it is used when we actually go and make predictions
# one epoch is how much time will we train for
# epoch is one run over our entire set of data

hist = model.fit(train, epochs=5, validation_data=val, callbacks=[tensorboard_callback])


# In[58]:


hist.history


# # 5. Plotting Performance

# In[59]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[60]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# # 6. Evaluation

# In[61]:


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[62]:


# metrics
pre = Precision()
re = Recall()
acc = BinaryAccuracy()


# In[63]:


for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


# In[64]:


print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')


# # 7. Testing

# In[65]:


import cv2


# In[65]:


# carrots (Recyclable) test
img = cv2.imread('carrot.jpg')
plt.imshow(img)
plt.show()


# In[66]:


# color fixing
img = cv2.imread('carrot.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# In[67]:


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[68]:


yhat = model.predict(np.expand_dims(resize/255, 0))


# In[69]:


yhat


# In[78]:


# correct prediction
# correct - recyclable


# In[70]:


if yhat <= 0.5: 
    print(f'Predicted class is Recyclable')
else:
    print(f'Predicted class is Non-Recyclable')


# In[87]:


# grass (Organic) test
img = cv2.imread('daal_nr.jpg')
plt.imshow(img)
plt.show()


# In[88]:


# color fixing
img = cv2.imread('daal_nr.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# In[89]:


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[90]:


yhat = model.predict(np.expand_dims(resize/255, 0))


# In[91]:


yhat


# In[92]:


# correct prediction
# correct - organic - non recyclable


# In[93]:


if yhat <= 0.5: 
    print(f'Predicted class is Recyclable')
else:
    print(f'Predicted class is Non-Recyclable')


# # 8. Saving the model

# In[78]:


from tensorflow.keras.models import load_model


# In[79]:


# .h is a serialization format
model.save(os.path.join('models','wastemodel.h5'))


# In[80]:


os.path.join('models','wastemodel.h5')


# In[81]:


new_model = load_model(os.path.join('models','wastemodel.h5'))


# In[82]:


new_model


# In[83]:


yhatnew = new_model.predict(np.expand_dims(resize/255, 0))

