import h5py
import numpy as np
import yaml
from math import ceil
import matplotlib.pyplot as plt

# load param configuration
with open('config/param.yml', 'r') as yml_file:
    cfg = yaml.safe_load(yml_file)

# open the hdf5 file
hdf5_path = cfg['train_hdf5']
hdf5_file = h5py.File(hdf5_path, "r")

# Total number of samples
train_img = hdf5_file["train_img"]
test_img = hdf5_file["test_img"]
train_num = train_img.shape[0]
test_num = test_img.shape[0]

# display image
img_num = cfg['demo_num']
test = test_img[100]
plt.imshow(test)
plt.show()
