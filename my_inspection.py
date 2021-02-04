"""
Retrain the YOLO model for your own dataset.
"""
import os, time, random, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import h5py
import tensorflow.keras.backend as K
from tensorflow.distribute import MirroredStrategy as multi_gpu_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TerminateOnNaN, LambdaCallback
from tensorflow_model_optimization.sparsity import keras as sparsity

from yolo5.model import get_yolo5_train_model
from yolo3.model import get_yolo3_train_model
from yolo3.data import yolo3_data_generator_wrapper, Yolo3DataGenerator
from yolo3.loss import yolo3_loss
from yolo2.model import get_yolo2_train_model
from yolo2.data import yolo2_data_generator_wrapper, Yolo2DataGenerator
from common.utils import get_classes, get_anchors, get_dataset, optimize_tf_gpu
from common.model_utils import get_optimizer
from common.callbacks import EvalCallBack, DatasetShuffleCallBack

# Try to enable Auto Mixed Precision on TF 2.0
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'

import tensorflow as tf
optimize_tf_gpu(tf, K)

# PARAMETERS
val_split = 0.3
batch_size = 16
model_image_size = (224, 224)
enhance_augment = None #/ "mosaic"
rescale_interval = 10
epochs = 5

# GET DATA
anchors = get_anchors("configs/yolo3_anchors.txt")
classes = get_classes("configs/epic_kitchens_classes.txt")
num_classes = len(classes)
num_anchors = len(anchors)
num_feature_layers = num_anchors // 3
model_type = "yolo3_vgg16"
weights_path = None
# weights = h5py.File("./my_model.h5", 'r')
dataset = get_dataset("epic_kitchens_train_data.txt")
num_val = int(len(dataset)*val_split)
num_train = len(dataset) - num_val
rescale_interval = -1

# SET UP LOGGING
logging = TensorBoard(log_dir="logs/001", histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
checkpoint = ModelCheckpoint(os.path.join(".", 'currweights.h5'),
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_weights_only=False,
    save_best_only=True,
    period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode='min', patience=10, verbose=1, cooldown=0, min_lr=1e-10)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min')
terminate_on_nan = TerminateOnNaN()
shuffle_callback = DatasetShuffleCallBack(dataset)
callbacks=[logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan, shuffle_callback]

# GET TRAINING MODELS AND DATA
train_model = get_yolo3_train_model(model_type, anchors, num_classes, model_pruning=False, weights_path=None)
optimizer = get_optimizer('adam', 0.001, decay_type=None)
train_data = yolo3_data_generator_wrapper(dataset[:num_train], batch_size, model_image_size, anchors, num_classes, enhance_augment, rescale_interval, multi_anchor_assign=False)
validation_data = yolo3_data_generator_wrapper(dataset[num_train:], batch_size, model_image_size, anchors, num_classes, multi_anchor_assign=False)

# # TRAIN
# train_model.fit(train_data, 
#                 steps_per_epoch=max(1, num_train//batch_size),
#                 validation_data=validation_data,
#                 validation_steps=max(1,num_val//batch_size),
#                 epochs=epochs,
#                 initial_epoch=1,
#                 workers=1,
#                 use_multiprocessing=False,
#                 max_queue_size=10,
#                 callbacks=callbacks)

# # quit()
# time.sleep(2)

# TRAIN
for i in range(len(train_model.layers)):
    train_model.layers[i].trainable = True
train_model.compile(optimizer=optimizer, loss=train_model.loss)
print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(num_train, num_val, batch_size, model_image_size))
train_model.fit(train_data, 
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=validation_data,
                validation_steps=max(1,num_val//batch_size),
                epochs=epochs,
                initial_epoch=1,
                workers=1,
                use_multiprocessing=False,
                max_queue_size=10,
                callbacks=callbacks)
train_model.save(os.path.join(".", 'my_model.h5'))
train_model.summary()
# print(yolo3_model)