import colorsys
import os, sys, argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import time
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Lambda
from tensorflow_model_optimization.sparsity import keras as sparsity
from PIL import Image

from yolo5.model import get_yolo5_model, get_yolo5_inference_model
from yolo5.postprocess_np import yolo5_postprocess_np
from yolo3.model import get_yolo3_model, get_yolo3_inference_model
from yolo3.postprocess_np import yolo3_postprocess_np
from yolo2.model import get_yolo2_model, get_yolo2_inference_model
from yolo2.postprocess_np import yolo2_postprocess_np
from common.data_utils import preprocess_image
from common.utils import get_classes, get_anchors, get_colors, draw_boxes, optimize_tf_gpu
from tensorflow.distribute import MirroredStrategy as multi_gpu_model

optimize_tf_gpu(tf, K)

image_filename = "./example_image.jpg"

class YOLO(object):

    def __init__(self, model_image_size, elim_grid_sense, classes_path, model_type,
                  anchors_path, weights_path):
        super(YOLO, self).__init__()
        # self.__dict__.update(kwargs) # and update with user overrides
        self.model_image_size = model_image_size
        self.elim_grid_sense = elim_grid_sense
        self.classes_path = classes_path
        self.model_type = model_type
        self.anchors_path = anchors_path
        self.weights_path = weights_path
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(self.class_names)
        K.set_learning_phase(0)
        self.yolo_model = self._generate_model()

    def _generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        num_feature_layers = num_anchors//3
        yolo_model, _ = get_yolo3_model(self.model_type, num_feature_layers, num_anchors, num_classes, input_shape=self.model_image_size + (3,), model_pruning=False)
        
        yolo_model.load_weights(weights_path) # make sure model, anchors and classes match
        # except Exception as e:
        #     print(repr(e))
        #     assert yolo_model.layers[-1].output_shape[-1] == \
        #         num_anchors/len(yolo_model.output) * (num_classes + 5), \
        #         'Mismatch between model and given anchor and class sizes'
        print('{} model, anchors, and classes loaded.'.format(weights_path))
        return yolo_model


    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_image_size)
        #origin image shape, in (height, width) format
        image_shape = tuple(reversed(image.size))

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))

        left_empty, right_empty = len(np.where(out_classes == 0)[0]) == 0, len(np.where(out_classes == 1)[0]) == 0
        # print(len(np.where(out_classes == 0)[0]) == 0)
        # quit()
        # if np.where(out_classes == 1)[0] is []:
        #     print("Empty right array")
        #     quit()
        
        # Get maximum left and right hand confidence scores
        if not left_empty:
            left_scores = out_scores[np.where(out_classes == 0)]
            left_box = out_boxes[np.where(out_scores == np.max(left_scores))]
            if not right_empty:
                out_boxes[1,:] = left_box[0,:]
            else:
                out_boxes[0,:] = left_box[0,:]
        if not right_empty:
            right_scores = out_scores[np.where(out_classes == 0)]
            right_box = out_boxes[np.where(out_scores == np.max(right_scores))]
            out_boxes[0,:] = right_box[0,:]
        # right_scores = out_scores[np.where(out_classes == 1)]
        # right_box = out_boxes[np.where(out_scores == np.max(right_scores))]
        # out_boxes[0,:], out_boxes[1,:] = left_box[0,:], right_box[0,:]
        out_boxes = out_boxes[:2,:] if ((not left_empty) and (not right_empty)) else out_boxes[:1,:] if ((not left_empty) or (not right_empty)) else out_boxes
        if ((not left_empty) and (not right_empty)):
            out_boxes = out_boxes[:2,:]
            out_classes = np.array([0, 1])
            out_scores = np.array([np.max(left_scores), np.max(right_scores)])
        elif ((not left_empty) or (not right_empty)):
            out_boxes = out_boxes[:1,:]
            out_classes = np.array([1]) if left_empty else np.array([0])
            out_scores = np.array([np.max(right_scores)]) if left_empty else np.array([np.max(right_scores)])


        #draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)
        return Image.fromarray(image_array), len(out_boxes)

    def bounding_boxes(self, image):
        image = Image.open(image)
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        image_data = preprocess_image(image, self.model_image_size)
        #origin image shape, in (height, width) format
        image_shape = tuple(reversed(image.size))

        return self.predict(image_data, image_shape)

        
    def predict(self, image_data, image_shape):
        num_anchors = len(self.anchors)
        out_boxes, out_classes, out_scores = yolo3_postprocess_np(self.yolo_model.predict(image_data), image_shape, self.anchors, len(self.class_names), self.model_image_size, max_boxes=100, elim_grid_sense=self.elim_grid_sense)
        return out_boxes, out_classes, out_scores


    def dump_model_file(self, output_model_file):
        self.yolo_model.save(output_model_file)

model_image_size = (224, 224)
elim_grid_sense = False
classes_path = "configs/epic_kitchens_classes.txt"
model_type = "yolo3_vgg16"
anchors_path = "configs/yolo3_anchors.txt"
weights_path = "./my_model.h5"

# args = [model_image_size, elim_grid_sense, classes_path, model_type,
#         anchors_path, weights_path]

yolo_model = YOLO(model_image_size, elim_grid_sense, classes_path, model_type,
                  anchors_path, weights_path)

image = Image.open(image_filename)
r_image, num_boxes = yolo_model.detect_image(image)
if num_boxes > 0:
    r_image.show()
else:
    print("Didn't find any bounding boxes")
# quit()
