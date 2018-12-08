import pybullet as p
import numpy as np


def capture_raw(height, width, flags=0, **kwargs):
    """
    use flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX to get
    depth and semantic segmentation map
    :param height: height of the image
    :param width: width of the image
    :param flags: flags to use, default 0
    :return: a tuple of images
    """
    return p.getCameraImage(width, height, flags=flags, **kwargs)


def get_segmentation_mask_object_and_link_index(seg_image):
    """
    Following example from
    https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/segmask_linkindex.py
    :param seg_image: [H, W] segmentation bitmap
    :return: object id map
    :return: link id map
    """
    assert(seg_image.ndim == 2)
    bmask = seg_image >= 0
    obj_idmap = bmask.copy().astype(np.int64) - 1
    link_idmap = obj_idmap.copy() - 1
    obj_idmap[bmask] = seg_image[bmask] & ((1 << 24) - 1)
    link_idmap[bmask] = (seg_image[bmask] >> 24) - 1
    return obj_idmap, link_idmap


def get_depth_map(depth_image, far=1000., near=0.01):
    """
    compute a depth map given a depth image and projection frustrum
    https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
    :param depth_image:
    :param far: frustrum far range
    :param near: frustrum near range
    :return: a depth map
    """
    assert (depth_image.ndim == 2)
    depth = far * near / (far - (far - near) * depth_image)
    return depth


def get_images(height, width, flags, **kwargs):
    ims = capture_raw(height, width, flags, **kwargs)
    obj_idmap, link_idmap = get_segmentation_mask_object_and_link_index(ims[4])
    depth_map = get_depth_map(ims[3])
    return ims[2], obj_idmap, link_idmap, depth_map


def main():
    p.connect(p.GUI)
    id = p.loadURDF("../models/cup.urdf", [0, 0, 1], globalScaling=10.0)
    id = p.loadURDF("../models/cup.urdf", [0, 1, 1], globalScaling=10.0)
    id = p.loadURDF("../models/cup.urdf", [1, 1, 0], globalScaling=10.0)
    ims = capture_raw(240, 320)
    obj_idmap, link_idmap = get_segmentation_mask_object_and_link_index(ims[4])
    depth_map = get_depth_map(ims[3])


if __name__ == '__main__':
    main()
