import pygame as pg
#import numpy as np
from numba import njit, prange, jit
from model import *
from camera import *


pg.init()


def vec(*args):
    return np.array([*args])


# глобальные переменные
w, h = 500, 500

model = load_obj('object.obj')
cam_pos = vec(0, 0, 2)
cam_rot = vec(0, math.pi)

zbuffer = [-float('infinity')]*w*h

def update_camera():
    return camera(cam_pos, cam_rot, math.pi / 3, w, h)

screen = pg.display.set_mode((w, h))
clock = pg.time.Clock()

HW, HH = w // 2, h // 2
to_screen = np.array([
    [HW, 0, 0, 0],
    [0, -HH, 0, 0],
    [0, 0, 1, 0],
    [HW, HH, 0, 1]
])

# функции
@njit(fastmath = True)
def cross(a, b, c):
    ab = a - b
    bc = c - b

    return norm([ab[1] * bc[2] - ab[2] * bc[1], ab[2] * bc[0] - ab[0] * bc[2], ab[0] * bc[1] - ab[1] * bc[0]])
    #return ab * bc


@njit(fastmath = True)
def norm(v):
    s = max(1, (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5)

    return np.array([v[0] / s, v[1] / s, v[2] / s])


@njit(fastmath = True)
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(fastmath = True)
def get_light(n, ld):
    return max(0, min(1, dot(n, ld) * 0.8 + 0.2))


def triangle(color, *p):
    #print(p0, p1, p2)
    pg.draw.polygon(screen, color, [*p])


#@njit(parallel = True)
def draw_model(model, cam):
    vert = model[0] @ cam[0]
    vert = vert @ cam[1]
    vert = vert @ to_screen

    nv = []

    for v in vert:
        vert = v / v[3]
        vert[0] = 0 if vert[0] < 1 else vert[0]
        vert[1] = 0 if vert[1] < 1 else vert[1]
        vert[2] = 0 if vert[2] < 1 else vert[2]
        nv.append(vert)

    vert = nv

    for f in model[1]:
        n = cross(vert[f[0]], vert[f[1]], vert[f[2]])
        l = get_light(n, norm(vec(1, -1, 1)))

        if l > 0:
            points = []

            for v in f:
                vertex = vert[v]

                points.append(vertex[:2])

            triangle((255*l, 255*l, 255*l), *points)


# код
while True:
    clock.tick(60)

    for e in pg.event.get():
        if e.type == pg.QUIT:
            exit()

    screen.fill((0, 0, 0))

    model[0] = model[0] @ rotate_y(math.radians(1))

    #cam_rot[1] += math.radians(1)

    cam = update_camera()

    draw_model(model, cam)

    pg.display.flip()
    pg.display.set_caption(str(clock.get_fps()))
    zbuffer = [-float('infinity')]*w*h
