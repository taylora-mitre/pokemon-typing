from typing import Union, List

import numpy as np
import scipy as sp
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
import json


IMSHAPE_TYPE = tf.int32
IMAGE_TYPE = tf.int32
IMAGE_TYPE_numpy = 'int32'
IMSHAPE_TYPE_numpy = 'int32'

# Useful in assertions later
def isscalar(x):
    return np.size(x)==1
@tf.function(reduce_retracing=True)
def isscalar_tf(x):
    val = (tf.size(x) == 1)
    return val.numpy() if hasattr(val, 'numpy') else tf.cast(val, tf.bool)

###########################
# class: ImageResampler
###########################
class ImageResampler():
    def __init__(self, **kwargs):
        self.seed = kwargs.get('seed', 623)
        self.reset_random()

        # Order of operations, allows certain processes to be skipped
        # Do nothing by default
        self.pipeline = kwargs.get('pipeline', [])

        # Convert all pipeline args to tf Tensors
        #   where each row is like (opname, arg0)
        self.opnames = tf.convert_to_tensor([tf.convert_to_tensor(row[0], dtype=tf.string) for row in self.pipeline if row[0] != 'normalize'])
        self.args = tf.convert_to_tensor([tf.convert_to_tensor(row[1:], dtype=tf.float32) for row in self.pipeline if row[0] != 'normalize'])
        #self.pipeline = tf.data.Dataset.from_tensor_slices((opnames, args)) # do I need this?

        # Vars to save to easily reload resampler
        self.vars_to_save = ['seed', 'pipeline'] # keep pipeline, since it's more easily serializable

    # region RNG handling
    @staticmethod
    def generate_random_numpy(amplitude: float) -> float:
        """
        Generates a random number between +/- amplitude
        >>> amp = 10
        >>> all([-amp <= Resampler.generate_random_numpy(amp) <= amp for i in range(100)])
        True
        """
        return amplitude * (1-2*np.random.random())

    def reset_random(self):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
    # endregion

    # region import/export
    def to_dict(self):
        export_dict = {}
        for key in self.vars_to_save:
            val = getattr(self, key)

            # Need to make values serializable
            if isinstance(val, tf.Tensor):
                export_dict[key] = val.numpy().tolist() 
            elif isinstance(val, np.ndarray):
                export_dict[key] = val.tolist()
            elif hasattr(val, 'to_dict'):
                export_dict[key] = val.to_dict()
            else:
                export_dict[key] = val 

        return export_dict

    # Wrapper around convert_to_dict
    # Subclasses should not overwrite this
    def export(self, file):
        dkt = self.to_dict()
        with open(file, 'w') as fopen:
            json.dump(dkt, fopen)

    @staticmethod
    def load(x):
        if isinstance(x, str): # read in from JSON file
            with open(x, 'r') as fopen:
                dkt = json.load(fopen)
        elif isinstance(x, dict):
            dkt = x.copy()
        
        # Make this easier to integrate via super() calls
        return dkt
    
    # endregion

    # region TF functions
    @staticmethod
    @tf.function
    def ceiling_elem(img: tf.Tensor, ceiling_pct: tf.Tensor = tf.constant(1.0)) -> tf.Tensor:
        # Check valid input
        tf.Assert(isscalar_tf(ceiling_pct), ['Inter-quartile range must be a scalar:', ceiling_pct])
        
        # Convert to 0 < IQR < 1
        if ceiling_pct > 1:
            ceiling_pct = tf.cast(ceiling_pct, tf.float32) / 100.0

        # Clip the image by value
        ceiling = ImageResampler._get_quantile(img, ceiling_pct)
        new_img = tf.clip_by_value(img, tf.math.reduce_min(img), ceiling)
        return new_img

    @staticmethod
    @tf.function(reduce_retracing=True)
    def crop_elem(img: tf.Tensor, shape: tf.Tensor) -> tf.Tensor:
        img = tf.squeeze(img)

        # Format input
        shape = tf.cast(shape, IMSHAPE_TYPE)
        imshape = tf.cast(tf.shape(img), IMSHAPE_TYPE)

        # Value == 999 means do something new
        if isscalar_tf(shape) and shape == tf.constant(999):
            tf.print('ERROR! Not sure what to do here inside 2d crop:', shape)
            shape = tf.convert_to_tensor([64,64], dtype=IMSHAPE_TYPE)
        # Value == 0 means do nothing
        elif isscalar_tf(shape) and shape == tf.constant(0):
            shape = imshape
        # This is probably what I'll use the most?
        elif isscalar_tf(shape):
            shape = tf.convert_to_tensor([shape, shape], dtype=IMSHAPE_TYPE)
        # Provided shape is valid as-is
        else:
            pass

        # Check valid input
        tf.Assert(shape[0] <= imshape[0], ['Cannot crop to a larger first dim! Input shape:', imshape, '| Desired shape:', shape])
        tf.Assert(shape[1] <= imshape[1], ['Cannot crop to a larger first dim! Input shape:', imshape, '| Desired shape:', shape])

        # Establish index amount to crop out on both sides
        xdiff = tf.cast((imshape[1]-shape[1])/2, IMSHAPE_TYPE)
        ydiff = tf.cast((imshape[0]-shape[0])/2, IMSHAPE_TYPE)

        # Crop in each dim
        new_img = img[
            ydiff : ydiff + shape[0],
            xdiff : xdiff + shape[1]
        ]
        return new_img

    @staticmethod
    @tf.function(reduce_retracing=True)
    def floor_ceiling_elem(img: tf.Tensor, IQR: tf.Tensor = tf.constant(1.0)) -> tf.Tensor:
        # Check valid input
        tf.Assert(isscalar_tf(IQR), ['Inter-quartile range must be a scalar:', IQR])
        
        # Convert to 0 < IQR < 1
        if IQR > 1:
            IQR = tf.cast(IQR, tf.float32) / 100.0

        # Clip the image by value
        floor   = ImageResampler._get_quantile(img, 0.5-IQR/2)
        ceiling = ImageResampler._get_quantile(img, 0.5+IQR/2)
        new_img = tf.clip_by_value(img, floor, ceiling)
        return new_img

    @staticmethod
    @tf.function(reduce_retracing=True)
    def normalize_elem(img: tf.Tensor) -> tf.Tensor:
        img_float = tf.cast(img, tf.float32)
        new_img = (img_float - tf.math.reduce_mean(img_float)) / (tf.math.reduce_std(img_float) + 0.1)
        return new_img

    @staticmethod
    @tf.function(reduce_retracing=True)
    def pad_elem(img: tf.Tensor, shape: tf.Tensor) -> tf.Tensor:
        # Format input
        shape = tf.cast(shape, IMSHAPE_TYPE) # use int shapes

        # Value == 999 means do something new
        if isscalar_tf(shape) and shape == tf.constant(999):
            BIG_SHAPE = tf.constant([512, 512]) # enormous just cuz
            shape = tf.cast(BIG_SHAPE, IMSHAPE_TYPE)

        # Value == 0 means do nothing
        elif isscalar_tf(shape) and shape == tf.constant(0):
            shape = tf.shape(img)
        
        # This is probably what I'll use the most?
        elif isscalar_tf(shape):
            shape = tf.convert_to_tensor([shape, shape, shape], dtype=IMSHAPE_TYPE)

        else:
            shape = tf.cast(shape, IMSHAPE_TYPE)

        # Compute total pads and split L/R
        ypad,xpad = tf.unstack(shape - tf.shape(img), num=2)
        ypad,xpad = tf.math.reduce_max([ypad,0]), tf.math.reduce_max([xpad,0]) # if shape already larger than pad, just pass
        yL, xL = tf.cast(ypad/2, IMSHAPE_TYPE), tf.cast(xpad/2, IMSHAPE_TYPE)
        yR, xR = ypad-yL, xpad-xL

        # Perform the operation and return
        padded = tf.pad(img, [[yL, yR], [xL, xR]])
        return padded

    @staticmethod
    @tf.function(reduce_retracing=True)
    def recenter_elem(img: tf.Tensor, center_yx: tf.Tensor = tf.constant(0)) -> tf.Tensor:
        # Assume "center" enters as [y,x], like imshape

        # Store current center
        imshape_center = tf.cast(tf.shape(img)/2, IMSHAPE_TYPE) # e.g. 3x3 returns [1,1] and 4x4 returns [2,2] 

        # Value == 999 means do something new
        if isscalar_tf(center_yx) and center_yx == tf.constant(999):
            tf.print('ERROR! Not sure what to do here in recenter2d')
            center_yx = tf.convert_to_tensor([32,32], dtype=IMSHAPE_TYPE)
        # Value == 0 means do nothing
        elif isscalar_tf(center_yx) and center_yx == tf.constant(0):
            center_yx = imshape_center
        # e.g. 32 --> [32, 32] -- likely never use this
        elif isscalar_tf(center_yx):
            center_yx = tf.convert_to_tensor([center_yx, center_yx], dtype=IMSHAPE_TYPE)
        # Most often -- pass a center to use
        else:
            center_yx = tf.cast(center_yx, IMSHAPE_TYPE)
        
        # Check valid input
        tf.Assert(len(center_yx)==2, ['Center is not a len-2 tensor:', center_yx])

        # Perform the operation and return -- pad in both dims then crop
        # If new_center > imshape_center, need to pad to the right -- else pad left
        ypad,xpad = tf.unstack(2*tf.cast(center_yx - imshape_center, IMSHAPE_TYPE), num=2)
        yleft,xleft = tf.math.reduce_max([0,-ypad]), tf.math.reduce_max([0,-xpad])
        yright,xright = tf.math.reduce_max([0,ypad]), tf.math.reduce_max([0,xpad])

        padded = tf.pad(img, [[yleft, yright], [xleft, xright]])

        new_img = ImageResampler.crop_elem(padded, tf.shape(img))

        return new_img

    # endregion

    # region numpy functions

    @staticmethod
    def shift_elem(img: np.ndarray, shift_xy: np.ndarray) -> np.ndarray:
        # This will be called from numpy functions (b/c need randomness), 
        #   so this may as well be a numpy function
        shift_xyz = np.append(shift_xy, 0) # add RGB channel dim, no shift here
        new_img = sp.ndimage.shift(img, shift_xyz, order=3)
        return new_img.astype(IMAGE_TYPE_numpy)

    @staticmethod
    def rotate_elem(img: np.ndarray, angle: float) -> np.ndarray:
        # This will be called from numpy functions (b/c need randomness), 
        #   so this may as well be a numpy function

        # Skip gate
        if angle == 0: return img

        new_img = sp.ndimage.rotate(img, angle, reshape=False, order=3)
        return new_img.astype(IMAGE_TYPE_numpy)

    @staticmethod
    def zoom_elem(img: np.ndarray, scale_factor: float) -> np.ndarray:
        # This will be called from numpy functions (b/c need randomness), 
        #   so this may as well be a numpy function

        # This function zooms by enlarging the image from (d,d) to scale_factor*(d,d)
        img_enlarged = sp.ndimage.zoom(img, scale_factor)

        # So, we have to then crop by the difference of (BIG) minus (small)
        (y,x), (Y,X) = img.shape, img_enlarged.shape
        ydiff, xdiff = [int((A-a)/2) for a,A in zip([y,x], [Y,X])]
        new_img = img_enlarged[
            ydiff : ydiff+y,
            xdiff : xdiff+x
        ]

        return new_img.astype(IMAGE_TYPE_numpy)

    @staticmethod
    def track_imshape(img_og: np.ndarray, opnames: tf.Tensor, args: tf.Tensor) -> List[np.ndarray]:
        # Just keeps track of image shape through a "forward pass" of the resampling
        imshapes = [np.array(img_og.shape).astype(IMSHAPE_TYPE_numpy)]

        # Start with initial image
        imshape = np.array(img_og.shape)
        for opname, arg_list in zip(opnames, args): # tf.Tensors
            arg = arg_list[0]

            # Crop sets the imshape exactly
            if opname == 'crop':
                imshape = np.array([arg.numpy(), arg.numpy()])

            # Pad sets a floor of an imshape
            elif opname == 'pad':
                min_imshape = np.array([arg.numpy(), arg.numpy()])
                imshape = np.array([min(min_imshape[i], imshape[i]) for i in range(2)])
            
            # some ops do not change imshape
            elif opname in ['ceiling', 'floor_ceiling', 'normalize', 'recenter', 'relabel', 'undo_relabel', 'shift', 'rotate', 'zoom']:
                pass

            # Debug
            else:
                tf.print('Opname', opname, 'not recognized in 2D imshape tracking')
        
            imshapes.append(imshape.astype(IMSHAPE_TYPE_numpy))
        
        return imshapes

    # endregion

    @staticmethod
    @tf.function(reduce_retracing=True)
    def _get_quantile(arr, quantile): 
        quantile = tf.cast(quantile, tf.float32)
        if quantile > 1: quantile = quantile / 100
        if quantile == 1: return tf.cast(tf.math.reduce_max(arr), arr.dtype)
        arr_sorted = tf.sort(tf.reshape(arr, (-1,)))
        return arr_sorted[int(tf.cast(quantile, tf.float32) * tf.cast(len(arr_sorted), tf.float32))]

    @tf.function(reduce_retracing=True)
    def apply_op(self, ds, opname, arg):

        if opname == 'crop':
            ds = ds.map(lambda img, tpe: (self.crop_elem(img, arg), tpe))
        elif opname == 'ceiling':
            ds = ds.map(lambda img, tpe: (self.ceiling_elem(img, arg), tpe))
        
        # TODO finish this

    @tf.function
    def apply(self, ds):
        # TODO write this
        pass

    def apply_to_one_img():
        # TODO write this
        pass 

    def apply_to_one_img_inference():
        # TODO write this
        pass

