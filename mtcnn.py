import os,sys
import keras_mt_shared_cnn
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import load_model
import math
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import argparse
import keras
from sklearn import preprocessing
from keras import backend as K


K.set_session(
    K.tf.Session(
    config=K.tf.ConfigProto(
    intra_op_parallelism_threads=62,
    inter_op_parallelism_threads=62))
)


parser = argparse.ArgumentParser()

parser.add_argument( '--n_folds', type= int, default= 2, help= 'fold' )
parser.add_argument( '--fold', type= int, default= 0, help= 'fold' )
parser.add_argument( '--tasks', type= str, default= [ 'site' , 'grade', 'behavior', 'laterality', 'histology' ], help= 'fold' )
parser.add_argument('--data_size',type = int, default = 47542, help = "size of data")
parser.add_argument('--batch_size',type=int,default= 64, help = "this is self explanatory !!")
parser.add_argument('--samples',type=int,default=1000, help = "number of samples for Louisiana generator")
parser.add_argument('--filename',type=int,default=2,help="dont ask")
args = parser.parse_args()

y_actual = []
y_pred = []




def main():
    train_x = np.load( 'data/train_X.npy' )
    train_y = np.load( 'data/train_Y.npy' )
    test_x = np.load( 'data/test_X.npy' )
    test_y = np.load( 'data/test_Y.npy' )

    for task in range( 4 ):
        le = preprocessing.LabelEncoder()    
        le.fit( train_y[ :, task ] )
        train_y[ :, task ] = le.transform( train_y[ :, task ] )
        test_y[ :, task ] = le.transform( test_y[ :, task ] )

    max_vocab = np.max( train_x )
    max_vocab2 = np.max( test_x )
    if max_vocab2 > max_vocab:
        max_vocab = max_vocab2

    print( 'max_vocab:', max_vocab )


    np.random.seed( 0 )

    wv_len = 50
    wv_mat = np.random.randn( max_vocab + 1, wv_len).astype( 'float32' ) * 0.1
    #num_classes = np.max( train_y ) + 1

    num_classes = []
    num_classes.append(np.max( train_y[:,0] ) + 1)
    num_classes.append(np.max( train_y[:,1] ) + 1)
    num_classes.append(np.max( train_y[:,2] ) + 1)
    num_classes.append(np.max( train_y[:,3] ) + 1)


    #Adjust learning rate based on number of GPUs
    optimizer = keras.optimizers.Adadelta(lr = 1.0)

    cnn = keras_mt_shared_cnn.init_export_network(
        num_classes= num_classes,
        in_seq_len= 1500,
        vocab_size= len( wv_mat ),
        wv_space= wv_len,
        filter_sizes= [ 3, 4, 5 ],
        num_filters= [ 300, 300, 300 ],
        concat_dropout_prob = 0.5,
        emb_l2= 0.001,
        w_l2= 0.01,
        optimizer= optimizer)

    print( cnn.summary() )

    validation_data = ( { 'Input': test_x },
        { 'Dense0': test_y[ :, 0 ],
          'Dense1': test_y[ :, 1 ],
          'Dense2': test_y[ :, 2 ],
          'Dense3': test_y[ :, 3 ] } )

    history = cnn.fit( 
        x= np.array( train_x ),
        y= [ np.array( train_y[ :, 0 ] ), 
             np.array( train_y[ :, 1 ] ), 
             np.array( train_y[ :, 2 ] ),
             np.array( train_y[ :, 3 ] ) ],
        batch_size= 16, 
        epochs= 10,
        verbose= 2,
        validation_data= validation_data
    )
   
    history.history[ 'val_loss' ] = np.mean( history.history[ 'val_loss' ] ) 
    print( history.history )

if __name__ == "__main__":
    main()


