from __future__ import division, print_function, unicode_literals
import os
import cv2
from collections import defaultdict
from scipy.misc import imresize
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
from random import sample


width = 299
height = 299
channels = 3
INCEPTION_PATH = os.path.join("datasets", "inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")

# Get all the images as dictionary
PERSON_PATH = os.path.join("datasets", "person")
person_root_path = os.path.join(PERSON_PATH, "person_photos")
person_classes=os.listdir(person_root_path)
image_paths = defaultdict(list)

for person_class in person_classes:
    image_dir = os.path.join(person_root_path, person_class)
    for filepath in os.listdir(image_dir):
        if filepath.endswith(".jpg") or filepath.endswith(".png"):
            image_paths[person_class].append(os.path.join(image_dir, filepath))
            
for paths in image_paths.values():
    paths.sort()  
       

def prepare_image(example_image):
    image = imresize(example_image, (width, height))  
    return image.astype(np.float32)/255    

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
training = tf.placeholder_with_default(False, shape=[])

with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=training)

inception_saver = tf.train.Saver()

prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])

n_outputs = len(person_classes)

with tf.name_scope("new_output_layer"):
    person_logits = tf.layers.dense(prelogits, n_outputs, name="person_logits")
    Y_proba = tf.nn.softmax(person_logits, name="Y_proba")
    
y = tf.placeholder(tf.int32, shape=[None])


with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=person_logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    flower_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="person_logits")
    training_op = optimizer.minimize(loss, var_list=flower_vars)
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(person_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

person_class_ids = {flower_class: index for index, flower_class in enumerate(person_classes)}
print(person_class_ids) 

person_paths_and_classes = []
for person_class, paths in image_paths.items():
    for path in paths:
        person_paths_and_classes.append((path, person_class_ids[person_class]))   
        
test_ratio = 0.2
train_size = int(len(person_paths_and_classes) * (1 - test_ratio))

np.random.shuffle(person_paths_and_classes)

person_paths_and_classes_train = person_paths_and_classes[:train_size]
person_paths_and_classes_test = person_paths_and_classes[train_size:]

def prepare_batch(person_paths_and_classes, batch_size):
    batch_paths_and_classes = sample(person_paths_and_classes, batch_size)
    images = []
    for path, labels in batch_paths_and_classes:
        if(cv2.imread(path) is not None):
            images.append(cv2.imread(path)[:, :, :channels])
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors ranging from -1 to 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch   

X_batch, y_batch = prepare_batch(person_paths_and_classes_train, batch_size=4)
print(X_batch.shape,y_batch.shape,X_batch.dtype,y_batch.dtype)  

X_test, y_test = prepare_batch(person_paths_and_classes_test, batch_size=len(person_paths_and_classes_test))

# Training of Inception_v3 on Human Dataset uncomment below lines for training and comment lines below Testing step 

n_epochs = 50
batch_size = 40
n_iterations_per_epoch = len(person_paths_and_classes_train) // batch_size
'''
with tf.Session() as sess:
    init.run()
    inception_saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)

    for epoch in range(n_epochs):
        print("Epoch", epoch, end="")
        for iteration in range(n_iterations_per_epoch):
            print(".", end="")
            X_batch, y_batch = prepare_batch(person_paths_and_classes_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print("  Train accuracy:", acc_train)

        save_path = saver.save(sess, "./human_classification_checkpoints/person_model"+str(acc_train)+".ckpt")

 '''
# Testing the trained model and accuracy is 98% 

n_test_batches = 10
X_test_batches = np.array_split(X_test, n_test_batches)
y_test_batches = np.array_split(y_test, n_test_batches)

with tf.Session() as sess:
    saver.restore(sess, "./human_classification_checkpoints/person_model1.0.ckpt")

    print("Computing final accuracy on the test set (this will take a while)...")
    acc_test = np.mean([
        accuracy.eval(feed_dict={X: X_test_batch, y: y_test_batch})
        for X_test_batch, y_test_batch in zip(X_test_batches, y_test_batches)])
    print("Test accuracy:", acc_test)         
    
