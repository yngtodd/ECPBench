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

from hyperspace import hyperdrive
from hyperspace.kepler import load_results


parser = argparse.ArgumentParser()
parser.add_argument( '--n_folds', type= int, default= 2, help= 'fold' )
parser.add_argument( '--fold', type= int, default= 0, help= 'fold' )
parser.add_argument( '--tasks', type= str, default= [ 'site' , 'grade', 'behavior', 'laterality', 'histology' ], help= 'fold' )
parser.add_argument('--data_size',type = int, default = 47542, help = "size of data")
parser.add_argument('--batch_size',type=int,default= 64, help = "this is self explanatory !!")
parser.add_argument('--samples',type=int,default=1000, help = "number of samples for Louisiana generator")
parser.add_argument('--filename',type=int,default=2,help="dont ask")
parser.add_argument('--results_dir',type=str,default="",
                    help="dont ask")
args = parser.parse_args()

y_actual = []
y_pred = []


def objective(hparams):
    filter1 = int(hparams[0])
    filter2 = int(hparams[1])
    filter3 = int(hparams[2])
    num_filters1 = int(hparams[3])
    num_filters2 = int(hparams[4])
    num_filters3 = int(hparams[5])
    dropout = float(hparams[6])
    w_l2 = float(hparams[7])

    #Adjust learning rate based on number of GPUs
    optimizer = keras.optimizers.Adadelta(lr = 1.0)
    from keras import backend as K
    K.set_session(
	K.tf.Session(
	config=K.tf.ConfigProto(
	intra_op_parallelism_threads=62, 
	inter_op_parallelism_threads=62)))

    cnn = keras_mt_shared_cnn.init_export_network(
        num_classes= num_classes,
        in_seq_len= 1500,
        vocab_size= len( wv_mat ),
        wv_space= wv_len,
        filter_sizes= [ filter1, filter2, filter3 ],
        num_filters= [ num_filters1, num_filters2, num_filters3 ],
        concat_dropout_prob = dropout,
        emb_l2= 0.001,
        w_l2= w_l2,
        optimizer= optimizer)

    history = cnn.fit(
        x= np.array( train_x ),
        y= [ np.array( train_y[ :, 0 ] ),
             np.array( train_y[ :, 1 ] ),
             np.array( train_y[ :, 2 ] ),
             np.array( train_y[ :, 3 ] ) ],
        batch_size= 16,
        epochs= 3,
        verbose= 0,
        validation_data= validation_data
    )

    history.history[ 'val_loss' ] = np.mean( history.history[ 'val_loss' ] )
    print( history.history )

    return history.history['val_loss']


def main():
    global train_x
    global train_y
    global test_x
    global test_y
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
    
    global wv_len
    wv_len = 50
    global wv_mat
    wv_mat = np.random.randn( max_vocab + 1, wv_len).astype( 'float32' ) * 0.1
    #num_classes = np.max( train_y ) + 1

    global num_classes
    num_classes = []
    num_classes.append(np.max( train_y[:,0] ) + 1)
    num_classes.append(np.max( train_y[:,1] ) + 1)
    num_classes.append(np.max( train_y[:,2] ) + 1)
    num_classes.append(np.max( train_y[:,3] ) + 1)

    global validation_data
    validation_data = ( { 'Input': test_x },
        { 'Dense0': test_y[ :, 0 ],
          'Dense1': test_y[ :, 1 ],
          'Dense2': test_y[ :, 2 ],
          'Dense3': test_y[ :, 3 ] } )

    space = [(1, 10), 
             (1, 10),
             (1, 10),
             (5, 500),
             (5, 500),
             (5, 500),
             (0.0, 0.9),
             (0.000001, 0.1)]

    checkpoint = load_results(args.results_dir)

    hyperdrive(objective=objective,
               hyperparameters=space,
               results_path=args.results_dir,
               model="GP",
               n_iterations=11,
               checkpoints=True,
               verbose=True,
               restart=checkpoint,
               random_state=0)
    

if __name__ == "__main__":
    main()
