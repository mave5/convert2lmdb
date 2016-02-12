import numpy as np
import sys
import os

# find caffe root and import caffe
caffe_root = '/usr/local/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import lmdb
import mnist

# get numpy version
print np.__version__


# load mnist databse
mnist_path = caffe_root+"data/mnist"
print('loading train dataset ...')
images_train,labels_train=mnist.load_mnist("training",np.arange(10),mnist_path)
print images_train.shape
print labels_train.shape

# dataset dimensions
N =images_train.shape[0]
hight=images_train.shape[1]
width=images_train.shape[2]

# reshape to 4-D array
X=images_train.reshape(N,1,hight,width)
print images_train.shape

# labels
y = labels_train


# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size = X.nbytes *10

# open lmdb file to write
env = lmdb.open('/home/exx/Desktop/caffetour/mnist/mnist_train_lmdb/', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tostring()
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())



# load mnist databse
mnist_path = caffe_root+"data/mnist"
print('loading test dataset ...')
images_test,labels_test=mnist.load_mnist("testing",np.arange(10),mnist_path)
print images_test.shape
print labels_test.shape

# dataset dimensions
N =images_test.shape[0]
hight=images_test.shape[1]
width=images_test.shape[2]

# reshape to 4-D array
X=images_test.reshape(N,1,hight,width)
print images_test.shape

# labels
y = labels_test

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size = X.nbytes *10

# open lmdb file to write
env = lmdb.open('/home/exx/Desktop/caffetour/mnist/mnist_test_lmdb/', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tostring()
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())


















