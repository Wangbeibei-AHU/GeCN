from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import os
from utils import *
from models import GeCN
import numpy as np

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'Cora', 'Dataset string.')  # 'Cora', 'Citeseer'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.6, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')


def main(pre, n):
    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset, pre, n)
    # Some preprocessing
    features = preprocess_features(features)
    support = preprocess_adj(adj)

    # Define placeholders
    placeholders = {
        'support': tf.placeholder(tf.float32, shape=[features.shape[0],features.shape[0]]),
        'features': tf.placeholder(tf.float32, shape=features.shape),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    # Create model
    model = GeCN(placeholders, input_dim=features.shape[1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders,flag=0):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []
    test_acc_list = []
    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        
        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)
       
        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            break

    print("Optimization Finished!")
    
    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))    
    return test_acc


if __name__ == "__main__":    
    record=np.zeros([3,5])
    for pre in range(1, 4):
        for n in range(5):
            record[pre-1][n]=main(pre, n)
    print("10%'s result:",np.mean(record[0])," + ",np.std(record[0]))
    print("20%'s result:",np.mean(record[1])," + ",np.std(record[1]))
    print("30%'s result:",np.mean(record[2])," + ",np.std(record[0]))