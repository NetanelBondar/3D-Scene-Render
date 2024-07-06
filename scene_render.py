
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from numba import njit, float64
from numba.experimental import jitclass

spec = [
    ('position', float64[:]),
    ('radius_x', float64),
    ('radius_z', float64),
    ('height', float64),
    ('color', float64[:])]

@jitclass(spec)
class Cylinder:
    def __init__(self, position: np.array, radius_x: float, radius_z: float, height: float, color: np.array):
        self.position = position
        self.radius_x = radius_x
        self.radius_z = radius_z
        self.height = height
        self.color = color


IMAGE_SIZE = (1080, 1920)
IMAGE_BACKGROUND = (0.1, 0.1, 0.1)

def get_mapping_scalar(local_top_left: Tuple[float, float], local_bottom_right: Tuple[float, float],
                       global_top_left: Tuple[int, int], global_bottom_right: Tuple[int, int]) -> Tuple[int, int]:
    """
    calculates scale of x and y
    :param local_top_left: top left point of transformed given data
    :param local_bottom_right: bottom right point of transformed given data
    :param global_top_left: top left point of bounding box
    :param global_bottom_right: bottom right point of bounding box
    :return: scale of x and y
    """
    scale_x = (global_bottom_right[0] - global_top_left[0]) / (local_bottom_right[0] - local_top_left[0])
    scale_y = (global_bottom_right[1] - global_top_left[1]) / (local_bottom_right[1] - local_top_left[1])

    return int(scale_x), int(scale_y)

def given_to_real(cord_1: float, cord_2: float, cord_3: float, scale: int,
                  min_x_global: int, min_y_global: int, min_x: float, min_y: float) -> Tuple[int, int, int]:
    """
    converts transformed coordinates to pixel coordinates on image
    :param cord_1: first transformed coordinate
    :param cord_2: second transformed coordinate
    :param cord_3: third transformed coordinate
    :param scale: the scale to convert given coordinates to pixel coordinates
    :param min_x_global: min pixel column
    :param min_y_global: min pixel row
    :param min_x: min x coordinate
    :param min_y: min y coordinate
    :return: the three pixel coordinates as pixel coordinates on image
    """
    j = min_y_global + (cord_1 - min_x) * scale
    i = min_x_global + (cord_2 - min_y) * scale
    return int(i), IMAGE_SIZE[0] - int(j), int(cord_3)

@njit
def draw_body(cyl: Cylinder, pos_on_img: List, scale: int,
              norm_light: np.ndarray, img: np.ndarray):
    """
    draws the square body of a cylinder (without top and bottom ellipses)
    :param cyl: the cylinder
    :param pos_on_img: cylinder's pixel coordinates on image
    :param scale: the scale to convert given coordinates to pixel coordinates
    :param norm_light: light direction normalized
    :param img: the image to be colored
    """
    top_left_x = int(pos_on_img[0] - (cyl.radius_x * scale))
    top_left_y = int(pos_on_img[1] - ((cyl.height / 2) * scale))

    bottom_right_x = int(pos_on_img[0] + (cyl.radius_x * scale))
    bottom_right_y = int(pos_on_img[1] + ((cyl.height / 2) * scale))

    for i in range(top_left_y, bottom_right_y):
        for k in range(top_left_x, bottom_right_x):
            delta_x = (top_left_x + bottom_right_x) / 2 - k
            delta_y = 0
            delta_z = np.sqrt((cyl.radius_x * scale) ** 2 - delta_x ** 2 - delta_y ** 2)
            vector = np.array([delta_x, delta_y, delta_z])
            norm_vector = vector / np.linalg.norm(vector)
            result = np.dot(norm_light, norm_vector)
            img[i, k] = cyl.color * (0.05 + 0.95 * np.array((0, result)).max())

@njit
def draw_top_ellipse(cyl: Cylinder, pos_on_img: List, scale: int,
                     norm_light: np.ndarray, img: np.ndarray):
    """
    draws the top ellipse of a cylinder
    :param cyl: the cylinder
    :param pos_on_img: cylinder's pixel coordinates on image
    :param scale: the scale to convert given coordinates to pixel coordinates
    :param norm_light: light direction normalized
    :param img: the image to be colored
    """
    center_x = pos_on_img[0]
    center_y = int(pos_on_img[1] - ((cyl.height / 2) * scale))
    small = cyl.radius_z * scale
    big = cyl.radius_x * scale

    for i in range(IMAGE_SIZE[0]):
        for k in range(IMAGE_SIZE[1]):
            if ((i - center_y) / small) ** 2 + ((k - center_x) / big) ** 2 <= 1:
                vector = np.array([0, 0, 1], dtype=float64)
                result = np.dot(norm_light, vector)
                img[i, k] = cyl.color * (0.05 + 0.95 * np.array((0, result)).max())

@njit
def draw_bottom_ellipse(cyl: Cylinder, pos_on_img: List, scale: int,
                        norm_light: np.ndarray, img: np.ndarray):
    """
    draws the bottom ellipse of a cylinder
    :param cyl: the cylinder
    :param pos_on_img: cylinder's pixel coordinates on image
    :param scale: the scale to convert given coordinates to pixel coordinates
    :param norm_light: light direction normalized
    :param img: the image to be colored
    """
    center_x = pos_on_img[0]
    center_y = int(pos_on_img[1] + ((cyl.height / 2) * scale))
    small = cyl.radius_z * scale
    big = cyl.radius_x * scale

    top_left_x = int(pos_on_img[0] - ((cyl.radius_x / 2) * scale))
    bottom_right_x = int(pos_on_img[0] + ((cyl.radius_x / 2) * scale))

    for i in range(IMAGE_SIZE[0]):
        for k in range(IMAGE_SIZE[1]):
            if ((i - center_y) / small) ** 2 + ((k - center_x) / big) ** 2 <= 1:
                delta_x = (top_left_x + bottom_right_x) / 2 - k
                delta_y = 0
                delta_z = np.sqrt(big ** 2 - delta_x ** 2 - delta_y ** 2)
                vector = np.array([delta_x, delta_y, delta_z])
                norm_vector = vector / np.linalg.norm(vector)
                result = np.dot(norm_light, norm_vector)
                img[i, k] = cyl.color * (0.05 + 0.95 * np.array((0, result)).max())

def show_img(img: np.array, title: str):
    """
    shows the given image
    :param img: np array of pixels
    :param title: title of image
    """
    plt.title(title)
    plt.imshow(np.sqrt(img))
    plt.axis(False)
    plt.show()

def cam_dir_norm(cam_z_dir_norm=np.array([0.253, 0.87, 0.423])) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :return: camera's directional axis normalized
    """
    c = (cam_z_dir_norm[0] ** 2 + cam_z_dir_norm[1] ** 2) / cam_z_dir_norm[2]

    cam_y_dir = np.array([-cam_z_dir_norm[0], -cam_z_dir_norm[1], c])
    cam_y_dir_norm = cam_y_dir / np.linalg.norm(cam_y_dir)

    cam_x_dir_norm = np.cross(cam_y_dir_norm, cam_z_dir_norm)

    return cam_x_dir_norm, cam_y_dir_norm, cam_z_dir_norm

def light_dir_norm(cam_axis_mat, light_dir=np.array([0.242, 0.053, 0.969])) -> np.ndarray:
    """
    calculates the light direction transformed and normalized
    :param cam_axis_mat: normalized direction vectors of camera
    :param light_dir: given light direction
    :return: transformed light direction normalized
    """
    light_dir_transformed = np.dot(cam_axis_mat, light_dir)

    return light_dir_transformed / np.linalg.norm(light_dir_transformed)

def transform_cyls(cylinders: np.ndarray, cam_axis_mat: np.ndarray):
    """
    transforms position and size of cylinders
    :param cylinders: list of Cylinders
    :param cam_axis_mat: camera's axis directions
    """

    for cylinder in cylinders:
        cylinder.position = np.dot(cam_axis_mat, cylinder.position)
        cylinder.radius_z = cylinder.radius_x * np.dot(cam_axis_mat[2], np.array((0, 0, 1)))
        cylinder.height = np.dot(cam_axis_mat[2], np.array((0, 1, 0)))


def bounding_box(cylinders: np.ndarray) -> Tuple[float, float, float, float]:
    """
    :param cylinders: the cylinders
    :return: min and max x, min and max y
    """
    min_x = min(cylinder.position[0] - cylinder.radius_x for cylinder in cylinders)
    max_x = max(cylinder.position[0] + cylinder.radius_x for cylinder in cylinders)
    min_y = min(cylinder.position[2] - cylinder.radius_z for cylinder in cylinders)
    max_y = max(cylinder.position[2] + cylinder.radius_z for cylinder in cylinders)

    return min_x, max_x, min_y, max_y

def draw_cyls(cylinders: np.ndarray, cyls_real_pos: np.ndarray, img: np.ndarray,
              scale: int, norm_light: np.ndarray):
    """
    draws the cylinders on image and show the final image

    :param cylinders: the cylinders
    :param cyls_real_pos: pixel positions of cylinders
    :param img: the image to be colored and drawn
    :param scale: the scale to convert coordinates to pixel coordinates
    :param norm_light: the light direction transformed and normalized
    """
    arr = [(i, j) for i, j in zip(cyls_real_pos, cylinders)]
    arr.sort(key=lambda x: x[0][2])

    for pos, cylinder in arr:
        draw_body(cylinder, pos, scale, norm_light, img)
        draw_top_ellipse(cylinder, pos, scale, norm_light, img)
        draw_bottom_ellipse(cylinder, pos, scale, norm_light, img)

    show_img(img, "Final Image")


def main():

    img = np.full((*IMAGE_SIZE, 3), IMAGE_BACKGROUND)

    show_img(img, 'Base Image')

    cam_x_dir_norm, cam_y_dir_norm, cam_z_dir_norm = cam_dir_norm()

    cam_axis_mat = np.array([cam_y_dir_norm, cam_x_dir_norm, cam_z_dir_norm])

    norm_light = light_dir_norm(cam_axis_mat)

    cylinders = np.array([
        Cylinder(np.array([0.28, 1.86, 0.5]), 0.5, 0.5, 1, np.array([0.8, 0.2, 0.25])),
        Cylinder(np.array([1.58, 0.66, 0.5]), 0.5, 0.5, 1, np.array([0.8, 0.41, 0.225])),
        Cylinder(np.array([-0.65, 0.75, 0.5]), 0.5, 0.5, 1, np.array([0.1, 0.3, 0.6])),
        Cylinder(np.array([0.14, -0.67, 0.5]), 0.5, 0.5, 1, np.array([0.2, 0.2, 0.3]))
    ])

    transform_cyls(cylinders, cam_axis_mat)

    min_x, max_x, min_y, max_y = bounding_box(cylinders)

    min_y_global = 200
    min_x_global = 750
    max_y_global = 1030
    max_x_global = 1380

    _, scale = get_mapping_scalar(local_top_left=(min_x, min_y),
                                  local_bottom_right=(max_x, max_y),
                                  global_top_left=(min_x_global, min_y_global),
                                  global_bottom_right=(max_x_global,max_y_global))

    cyls_real_pos = np.array([given_to_real(*cylinder.position, scale, min_x_global, min_y_global, min_x, min_y)
                              for cylinder in cylinders])

    draw_cyls(cylinders, cyls_real_pos, img, scale, norm_light)


if __name__ == '__main__':
    main()






