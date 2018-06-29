import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
from glob import glob
from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output_dir', default='output/', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--input_dir', default='', type=str,
                    help='The directory of input')
parser.add_argument('--mask_dir', default='', type=str,
                    help='The directory of mask')


if __name__ == "__main__":
    ng.get_gpus(1)
    args = parser.parse_args()
    model = InpaintCAModel()
    input_dir=args.input_dir
    images=glob(input_dir+"/*")
    masks=glob(mask_dir+"/*")


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)

        for image_path,mask_path in zip(images,masks):
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)

            assert image.shape == mask.shape

            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :]
            print('Shape of image: {}'.format(image.shape))

            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)


            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            print('Model loaded.')
            result = sess.run(output)
            cv2.imwrite(args.output_dir+image_path.split("/")[-1], result[0][:, :, ::-1])
