import tensorflow as tf
import numpy as np
import cv2
import argparse

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_size = (320, 240)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img = img / 255.
    img = np.expand_dims(img, 0)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("image_path", type=str)

    args = parser.parse_args()

    model_path = args.model_path
    image_path = args.image_path

    graph = tf.Graph()
    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())


    with graph.as_default():
        input = tf.placeholder(np.float32, shape=[1, 240, 320, 1])
        tf.import_graph_def(graph_def)

    img = preprocess_image(image_path)

    with tf.Session(graph = graph) as sess:
        output_tensor_descriptor = graph.get_tensor_by_name('import/superpoint/pred_tower0/descriptor/conv2/bn/FusedBatchNorm:0')
        output_tensor_detector = graph.get_tensor_by_name('import/superpoint/pred_tower0/detector/Reshape:0')

        input_img_tensor = graph.get_tensor_by_name('import/superpoint/image:0')
        output = sess.run([output_tensor_descriptor, output_tensor_detector], feed_dict = {input_img_tensor: img})
        descriptor_output = output[0]
        detector_output = output[1]
        print("descriptor_output")
        print(descriptor_output)

        print("detector_output")
        print(detector_output)

