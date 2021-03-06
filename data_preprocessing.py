from random import shuffle
import yaml
import glob
import numpy as np
import h5py
import cv2

# List images and the labels
with open('config/param.yml', 'r') as yml_file:
    cfg = yaml.safe_load(yml_file)

# load params
hdf5_path = cfg['train_hdf5']
train_path = cfg['train_path']
train_pos = cfg['train_percent']
test_pos = train_pos + cfg['test_percent']
backend = cfg['backend']
assert(test_pos == 1)

addrs = glob.glob(train_path)
labels = [1 if 'cat' in addr else 0 for addr in addrs]

# shuffle dataset
dataset = list(zip(addrs, labels))
shuffle(dataset)
addrs, labels = zip(*dataset)

# Divide the data into train and test
assert(len(addrs) == len(labels))
total_num = len(addrs)

train_addrs = addrs[:int(train_pos*total_num)]
train_labels = labels[:int(train_pos*total_num)]

test_addrs = addrs[int(train_pos*total_num):]
test_labels = labels[int(train_pos*total_num):]

assert(len(train_addrs) == len(train_labels))
assert(len(test_addrs) == len(test_labels))

train_num = len(train_addrs)
test_num = len(test_addrs)

# Creating a HDF5 file
# check the order of data and chose proper data shape to save images
if backend == 'th': # theano
    train_shape = (train_num, 3, 224, 224)
    test_shape = (test_num, 3, 224, 224)
elif backend == 'tf': # tensorflow
    train_shape = (train_num, 224, 224, 3)
    test_shape = (test_num, 224, 224, 3)

# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')

hdf5_file.create_dataset("train_img", train_shape, np.int8)
hdf5_file.create_dataset("test_img", test_shape, np.int8)

hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

hdf5_file.create_dataset("train_labels", (train_num,), np.int8)
hdf5_file["train_labels"][...] = train_labels
hdf5_file.create_dataset("test_labels", (test_num,), np.int8)
hdf5_file["test_labels"][...] = test_labels

# Load images and save them
# a numpy array to save the mean of the images
mean = np.zeros(train_shape[1:], np.float32)

# loop over train addresses
for i in range(train_num):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, train_num))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if backend == 'th':
        img = np.rollaxis(img, 2)
    # save the image and calculate the mean so far
    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(train_num)

# loop over test addresses
for i in range(test_num):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, test_num))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if backend == 'th':
        img = np.rollaxis(img, 2)
    # save the image
    hdf5_file["test_img"][i, ...] = img[None]

# save the mean and close the hdf5 file
hdf5_file["train_mean"][...] = mean
hdf5_file.close()
