import numpy as np
import pathlib
from PIL import Image


def load_image(filepath: str or pathlib.Path, image_mode: str) -> Image:
    """
    Loads image file from path
    :param image_mode: to which mode will the image be converted, default: RGB
    :param filepath: path to the target image
    :return: loaded image
    """
    fp = open(filepath, "rb")
    im = Image.open(fp).convert(image_mode)
    im.load()
    fp.close()
    return im


def resize_image_by_factor(image: Image, enlargement: float) -> Image:
    """
    Resizes an image (conserving it´s aspect ratio) using a multiplier
    :param image: the target image
    :param enlargement: how many times the image will be enlarged (or downsized)
    :return: the resized image
    """
    new_size = np.array(image.size) * enlargement
    new_image = image.resize(new_size)
    return new_image


def fit_image_to_tile(image: Image, new_size: tuple[int, int], fit_method_name: str) -> Image:
    """
    Makes an image fit to the desired tile size
    :param image: the target image
    :param new_size: the dimension of the new size in pixels
    :param fit_method_name: the way in which this image will be resized, can be "resize" or "crop"
    :return: the fitted image
    """
    # TODO: add this to an Enum
    fit_methods = {"resize": stretch_image_to_size, "crop": crop_image_to_size}
    fit_method = fit_methods[fit_method_name]
    return fit_method(image, new_size)


def stretch_image_to_size(image: Image, new_size: tuple[int, int]) -> Image:
    """
    Resizes an image by stretching it
    :param image: target image
    :param new_size: the dimension of the new size in pixels
    :return: the fitted image
    """
    return image.resize(new_size)


def resize_image_keep_ratio(image: Image, proposed_size: tuple[int, int]) -> Image:
    """
    Resize an image (conserving it´s aspect ratio) to a tile. Uses the smallest dimension to fit the borders
    :param image: target image
    :param proposed_size: the dimension of the new size in pixels
    :return: the fitted image
    """
    w, h = proposed_size
    W, H = image.size
    resize_factor = min(w / W, h / H)
    final_size = (int(W * resize_factor), int(H * resize_factor))
    return image.resize(final_size)


def crop_image_to_size(image: Image, new_size: tuple[int, int]) -> Image:
    """
    Extracts the center of an image by cropping it
    :param image: target image
    :param new_size: the dimension of the new size in pixels
    :return: the fitted image
    """
    final_image = resize_image_keep_ratio(image, new_size)
    width_original, height_original = final_image.size
    width_final, height_final = new_size

    top_left_point_coordinates = ((width_original - width_final) / 2, (height_original - height_final) / 2)
    bottom_right_point_coordinates = ((width_original + width_final) / 2, (height_original + height_final) / 2)
    coordinates = top_left_point_coordinates + bottom_right_point_coordinates
    return final_image.crop(coordinates)


def get_average_rgb_from_image(image: Image) -> tuple[int]:
    """
    Gets the average RGB color of an image
    :param image: target image
    :return: the average color in red, green, and blue
    """
    im = np.array(image)
    w, h, d = im.shape
    return tuple(np.average(im.reshape(w * h, d), axis=0))


def find_images_in_path(images_directory: str or pathlib.Path) -> list[Image]:
    """
    Finds all images in a path and loads them
    :param images_directory: where can the input files can be found
    :return: the loaded images
    """
    file_list = list(pathlib.Path(images_directory).glob('*[png|jpeg|jpg|PNG|JPEG|JPG]'))
    images = []
    for fp in file_list:
        img = load_image(fp)
        if img:
            images.append(img)
    return images
