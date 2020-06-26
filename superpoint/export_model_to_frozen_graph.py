import yaml
import argparse
import logging
from pathlib import Path

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
import tensorflow as tf  # noqa: E402

from superpoint.models import get_model  # noqa: E402
from superpoint.settings import EXPER_PATH  # noqa: E402


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('export_name', type=str)
    args = parser.parse_args()

    export_name = args.export_name
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config['model']['data_format'] = 'channels_last'

    checkpoint_path = Path(EXPER_PATH, export_name)

    with get_model(config['model']['name'])(
            data_shape={'image': [1, 240, 320, 1]},
            **config['model']) as net:

        net.load(str(checkpoint_path)) # defined in base_model.py, basically tf.train.Saver and saver.restore

        output_nodes = ["superpoint/pred_tower0/detector/Reshape", "superpoint/pred_tower0/descriptor/conv2/bn/FusedBatchNorm"]

        output_graph_def = tf.graph_util.convert_variables_to_constants(
                net.sess, 
                net.sess.graph_def,
                output_nodes
        )       

        output_graph = "frozen_graph" + export_name + ".pb"

        with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
