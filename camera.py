import numpy as np
from matrix import *


def camera(pos, rot, fov, w, h):
    pos = np.array([*pos, 1.0])
    forward_d = np.array([0, 0, 1, 1])
    up_d = np.array([0, 1, 0, 1])
    right_d = np.array([1, 0, 0, 1])
    forward_d, up_d, right_d = camera_pitch(forward_d, up_d, right_d, rot[0])
    forward_d, up_d, right_d = camera_yaw(forward_d, up_d, right_d, rot[1])
    h_fov = fov
    v_fov = h_fov * (h / w)
    near_plane = 0.1
    far_plane = 100

    near = near_plane
    far = far_plane
    right = math.tan(h_fov / 2)
    left = -right
    top = math.tan(v_fov / 2)
    bottom = -top

    m00 = 2 / (right - left)
    m11 = 2 / (top - bottom)
    m22 = (far + near) / (far - near)
    m32 = -2 * near * far / (far - near)
    proj = np.array([
        [m00, 0, 0, 0],
        [0, m11, 0, 0],
        [0, 0, m22, 1],
        [0, 0, m32, 0]
    ])

    return translate_matrix(pos) @ rotate_matrix(forward_d, up_d, right_d), proj


def camera_yaw(forward, up, right, angle):
    rotate = rotate_y(angle)
    forward = forward @ rotate
    right = right @ rotate
    up = up @ rotate

    return forward, up, right

def camera_pitch(forward, up, right, angle):
    rotate = rotate_x(angle)
    forward = forward @ rotate
    right = right @ rotate
    up = up @ rotate

    return forward, up, right

def translate_matrix(pos):
    x, y, z, w = pos
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [-x, -y, -z, 1]
    ])

def rotate_matrix(forward, up, right):
    rx, ry, rz, w = right
    fx, fy, fz, w = forward
    ux, uy, uz, w = up
    return np.array([
        [rx, ux, fx, 0],
        [ry, uy, fy, 0],
        [rz, uz, fz, 0],
        [0, 0, 0, 1]
    ])