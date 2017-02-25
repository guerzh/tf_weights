Note: Note: *not tested on the full test set*; so only useful for pedagogical purposes. Proceed with caution.

Homepage:
http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

With newer versions of TensorFlow, use myalexnet_forward_newtf.py.

With older versions, use myalexnet_forward.py

bvlc_alexnet.py and bvlc_alexnet.npy are generated using https://github.com/ethereon/caffe-tensorflow by converting the AlexNet weights/model from here:
https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet


myalexnet is the implementation of AlexNet in TensorFlow

Weights are available here:
http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy

Try myalexnet_forward.py for a version with a placeholder as the input (useful for training). Otherwise see myalexnet.py
