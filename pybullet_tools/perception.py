import pybullet as p
import numpy as np


def capture_raw(height, width, **kwargs):
    """
    use flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX to get
    depth and semantic segmentation map
    :param height: height of the image
    :param width: width of the image
    :param flags: flags to use, default 0
    :return: a tuple of images
    """
    return p.getCameraImage(width, height, **kwargs)


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


def get_depth_map(depth_image, near=0.01, far=100.):
    """
    compute a depth map given a depth image and projection frustrum
    https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
    :param depth_image:
    :param near: frustrum near range
    :param far: frustrum far range
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


class Camera(object):
    def __init__(self, height, width, fov=60, near=0.01, far=100., renderer=p.ER_TINY_RENDERER):

        aspect = float(width) / float(height)
        self._height = height
        self._width = width
        self._near = near
        self._far = far
        self._view_matrix = p.computeViewMatrix([0, 0, 1], [0, 0, 0], [1, 0, 0])
        self._projection_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near, farVal=far)
        self._renderer = renderer

    def set_pose(self, camera_pos, target_pos, up_vector):
        assert(len(camera_pos) == 3)
        assert(len(target_pos) == 3)
        assert(len(up_vector) == 3)
        self._view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)

    def set_pose_ypr(self, target_pos, distance, yaw, pitch, roll=0, up_axis=2):
        assert(len(target_pos) == 3)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target_pos,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=up_axis
        )

    def capture_raw(self):
        return p.getCameraImage(
            self._width,
            self._height,
            list(self._view_matrix),
            list(self._projection_matrix),
            renderer=p.ER_TINY_RENDERER
        )

    def capture_frame(self):
        width, height, rgb, depth, seg = self.capture_raw()
        obj_idmap, link_idmap = get_segmentation_mask_object_and_link_index(seg)
        depth_map = get_depth_map(depth, near=self._near, far=self._far)
        return width, height, rgb[:, :, :3], depth_map, obj_idmap, link_idmap


def main():
    p.connect(p.GUI)
    id = p.loadURDF("../models/cup.urdf", [0, 0, 1], globalScaling=10.0)
    id = p.loadURDF("../models/cup.urdf", [0, 1, 1], globalScaling=10.0)
    id = p.loadURDF("../models/cup.urdf", [1, 1, 0], globalScaling=10.0)
    c = Camera(480, 640)
    c.set_pose_ypr([0, 0, 0], 5, 45, -45)
    w, h, rgb, depth, obj, link = c.capture_frame()
    import matplotlib.pyplot as plt


if __name__ == '__main__':
    main()
