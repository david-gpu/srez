import moviepy.editor as mpe
import numpy as np
import numpy.random
import os.path
import scipy.misc
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def demo1(sess):
    """Demo based on images dumped during training"""

    # Get images that were dumped during training
    filenames = tf.gfile.ListDirectory(FLAGS.train_dir)
    filenames = sorted(filenames)
    filenames = [os.path.join(FLAGS.train_dir, f) for f in filenames if f[-4:]=='.png']

    assert len(filenames) >= 1

    fps        = 30

    # Create video file from PNGs
    print("Producing video file...")
    filename  = os.path.join(FLAGS.train_dir, 'demo1.mp4')
    clip      = mpe.ImageSequenceClip(filenames, fps=fps)
    clip.write_videofile(filename)
    print("Done!")
    
