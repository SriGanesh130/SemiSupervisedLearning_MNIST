'''
Denoising Auto-Encoder on MNIST Data
'''
from sklearn.utils import shuffle
import keras
import os
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def generate_batches(images_repo, labels, batch_size):

    print('generating batches')

    images_repo = shuffle(images_repo)

    for batch in range(0, int(len(images_repo)), batch_size):

        yield images_repo[batch: batch + batch_size], labels[batch: batch + batch_size]


learning_rate = 1e-4
epochs = 10
batch_size = 200
noise_factor = 0.5
is_train = False
save_path = "./model/"
(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.mnist.load_data()
inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')

conv1 = tf.layers.conv2d(inputs=inputs_, filters=16, kernel_size=(
    3, 3), padding='same', activation=tf.nn.relu)
maxpool1 = tf.layers.max_pooling2d(
    conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(
    3, 3), padding='same', activation=tf.nn.relu)
encoded = tf.layers.max_pooling2d(
    conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
upsample1 = tf.layers.conv2d_transpose(encoded, filters=32, kernel_size=(
    3, 3), strides=(2, 2), padding="same", activation=tf.nn.relu)
upsample2 = tf.layers.conv2d_transpose(upsample1, filters=16, kernel_size=(
    3, 3), strides=(2, 2), padding="same", activation=tf.nn.relu)

logits = tf.layers.conv2d_transpose(inputs=upsample2, filters=1, kernel_size=(
    3, 3), padding='same', activation=None)

decoded = tf.nn.relu(logits)

cost = tf.reduce_mean(tf.abs(targets_-decoded))

opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# saver.restore(sess, "./model/model.ckpt")

train_loss = []
validation_loss = []
epoc = []

generator_test = generate_batches(test_images, test_labels, 10000)
validation_set = [batch_test for i, batch_test in enumerate(generator_test)]
print("validation generated")
for e in range(epochs):
    if is_train:
        generator = generate_batches(train_images, train_labels, batch_size)
        print("train generated")
        for i, batch in enumerate(generator):
            imgs = np.expand_dims(batch[0], -1)
            noisy_imgs = imgs/255.0 + np.random.normal(0.0, 0.3, imgs.shape)
            imgs_test = np.expand_dims(validation_set[0][0], -1)
            noisy_imgs_test = imgs_test/255.0 + \
                np.random.normal(0.0, 0.3, imgs_test.shape)
            batch_cost, _, generated = sess.run([cost, opt, decoded], feed_dict={inputs_: noisy_imgs,
                                                                                 targets_: imgs/255.0})
            print(i)
            validation_cost = sess.run(
                [cost], feed_dict={inputs_: noisy_imgs_test, targets_: imgs_test/255.0})
        print("Epoch: {}/{}...".format(e+1, batch_cost), "Training loss: {:.4f}".format(
            batch_cost), "validation loss: {:.4f}".format(validation_cost[0]))
        train_loss.append(batch_cost)
        validation_loss.append(validation_cost[0])
        epoc.append(e+1)
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(np.squeeze(generated[0] * 255.0), cmap='gray')
        axarr[1].imshow(np.squeeze(imgs[0] * 255.0), cmap='gray')
        axarr[2].imshow(np.squeeze(noisy_imgs[0] * 255.0), cmap='gray')
        plt.show()
        plt.figure()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.scatter(epoc, train_loss, c='blue', label="train loss")
        plt.scatter(epoc, validation_loss, c='red', label="validation loss")
        plt.legend()
        plt.show()
        plt.figure()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(epoc, train_loss, c="blue", label="train")
        plt.plot(epoc, validation_loss, c="red", label="validation loss")
        plt.legend()
        plt.show()

        if e == 9:
            save_path = saver.save(
                sess=sess, save_path=save_path)
            print("Model saved in file: %s" % save_path)
            print("Model saved in file: %s" % save_path)

    else:
        generator = generate_batches(test_images, test_labels, batch_size)
        for i, batch in enumerate(generator):
            imgs = np.expand_dims(batch[0], -1)
            noisy_imgs = imgs/255.0 + np.random.normal(0.0, 0.3, imgs.shape)
            generated = sess.run([decoded], feed_dict={inputs_: noisy_imgs,
                                                       targets_: imgs/255.0})
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(np.squeeze(generated[0][0]), cmap='gray')
        axarr[0].set_title("generated")
        axarr[1].imshow(np.squeeze(imgs[0] * 255.0), cmap='gray')
        axarr[1].set_title("original")
        axarr[2].imshow(np.squeeze(noisy_imgs[0] * 255.0), cmap='gray')
        axarr[2].set_title("noisy-input")
        plt.show()
