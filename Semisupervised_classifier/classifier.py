import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import keras
from sklearn.utils import shuffle


def generate_batches(images_repo, labels_category, batch_size):
    '''
    args: images_repo, labels, batch_size
    functionality: shuffled the data and divided in to batches
    reutnrn: images_repo
    '''
    print('generating batches')
    images_repo, labels_category = shuffle(images_repo, labels_category)
    labels = keras.utils.to_categorical(
        labels_category, num_classes=None, dtype='float32')
    for batch in range(0, int(len(images_repo)), batch_size):
        yield images_repo[batch: batch + batch_size], labels[batch: batch + batch_size]


# hyperparameters
lr = 1e-3
epochs = 100
batch_size = 100
noise_factor = 0.5
is_train = True
(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.mnist.load_data()
encoder_model_path = "./path/to/trained/autoencoder.ckpt"
# inputs
inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
labels = tf.placeholder(tf.float32, (None, 10), name='lables')
# Extcting encoder from autoencoder


def model(inputs_, labels):
    conv1 = tf.layers.conv2d(inputs=inputs_, filters=16, kernel_size=(
        3, 3), padding='same', activation=tf.nn.relu, trainable=False)
    maxpool1 = tf.layers.max_pooling2d(
        conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(
        3, 3), padding='same', activation=tf.nn.relu, trainable=False)
    encoded = tf.layers.max_pooling2d(
        conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
    # upsample1 = tf.layers.conv2d_transpose(encoded,filters = 32, kernel_size = (3, 3), strides = (2, 2), padding= "same",activation=tf.nn.relu)
    # upsample2 = tf.layers.conv2d_transpose(upsample1, filters = 16, kernel_size = (3, 3), strides = (2, 2), padding= "same",activation=tf.nn.relu)
    # logits = tf.layers.conv2d_transpose(inputs=upsample2, filters=1, kernel_size=(3,3), padding='same', activation=None)
    # decoded = tf.nn.relu(logits)
    return encoded


# adding a softmax-classifier
encoded = model(inputs_, labels)
fc = tf.reshape(encoded, (-1, 7*7*32))
dense1 = tf.layers.dense(fc, 1024, activation=tf.nn.relu)
dropout2 = tf.layers.dropout(dense1, rate=0.25)
dense2 = tf.layers.dense(dropout2, 10, activation=tf.nn.softmax)

# loss function
loss = tf.nn.softmax_cross_entropy_with_logits(
    logits=dense2,
    labels=labels
)
cost = tf.reduce_mean(loss)
# optimization
opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# prediction
correct_pred = tf.equal(tf.argmax(dense2, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# loading pre-trained encoder model
saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[:4])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, encoder_model_path)
train_loss = []
validation_loss = []
epoc = []
generator_test = generate_batches(test_images, test_labels, 10000)
validation_set = [batch_test for i, batch_test in enumerate(generator_test)]
for e in range(epochs):
    if is_train:
        generator = generate_batches(train_images, train_labels, batch_size)
        for i, batch in enumerate(generator):
            imgs = np.expand_dims(batch[0], -1)
            Y = batch[1]
            imgs_test = np.expand_dims(validation_set[0][0], -1)
            labels_test = validation_set[0][1]
            # training
            batch_cost, acc, _ = sess.run([cost, accuracy, opt], feed_dict={inputs_: imgs/255.0,
                                                                            labels: Y/255.0})
            # validation
            validation_cost, val_acc, pred = sess.run([cost, accuracy, dense2], feed_dict={inputs_: imgs_test/255.0,
                                                                                           labels: labels_test/255.0})
        print("Epoch: {}/{}...".format(e+1, batch_cost),
              "Training loss: {:.4f}".format(batch_cost), "accuracy: {}".format(acc))
        print("validation loss: {}".format(validation_cost),
              "validation loss: {:.4f}".format(val_acc))
