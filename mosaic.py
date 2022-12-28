import itertools

import numpy as np
from PIL import Image
import pathlib
from tqdm import tqdm


def load_image(filepath):
    fp = open(filepath, "rb")
    im = Image.open(fp).convert(Mosaic.PICTURE_MODE)
    im.load()
    fp.close()
    return im


def resize_image_by_factor(image, enlargement):
    new_size = np.array(image.size) * enlargement
    new_image = image.resize(new_size)
    return new_image


def fit_image_to_tile(image, new_size, fit_method_name):
    fit_methods = {"resize": stretch_image_to_size, "crop": crop_image_to_size}
    fit_method = fit_methods[fit_method_name]
    return fit_method(image, new_size)


def stretch_image_to_size(image, new_size):
    return image.resize(new_size)


def resize_image_mantain_ratio(image, proposed_size):
    w, h = proposed_size
    W, H = image.size
    resize_factor = max(w/W, h/H)
    final_size = (int(W * resize_factor), int(H * resize_factor))
    return image.resize(final_size)


def crop_image_to_size(image, new_size):
    final_image = resize_image_mantain_ratio(image, new_size)
    width_original, height_original = final_image.size
    width_final, height_final = new_size

    top_left_point_coordinates = ((width_original - width_final) / 2, (height_original - height_final) / 2)
    bottom_right_point_coordinates = ((width_original + width_final) / 2, (height_original + height_final) / 2)
    coordinates = top_left_point_coordinates + bottom_right_point_coordinates
    return final_image.crop(coordinates)


def get_average_rgb_from_image(image):
    im = np.array(image)
    w, h, d = im.shape
    return tuple(np.average(im.reshape(w * h, d), axis=0))


class Mosaic:
    PICTURE_MODE = 'RGB'

    def __init__(self, input_photo_path, other_photos_path, tile_size, output_file, enlargement=1, fit_method="resize"):

        self.image_coordinates_in_order = []

        # Size of the tile image in pixels
        self.tile_image_size = tile_size

        # Read the target image, enlarge it, and
        target_image = self.process_target_image(input_photo_path, enlargement)
        self.mosaic_image_size = target_image.size  # Size of the final image in pixels
        self.mosaic_image_shape = self.get_total_number_of_tiles()  # mosaic shape (number of tiles in rows and columns)

        total_tiles = self.mosaic_image_shape[0] * self.mosaic_image_shape[1]
        color_size = 3
        self.target_img_rgb_colors = np.zeros(shape=(total_tiles, color_size))

        # Divide the target image and calculate the average color per tile
        self.subdivide_original_image(target_image)

        # Read the input images and make them smaller
        self.input_images = []
        self.process_input_images(other_photos_path, fit_method_name=fit_method)
        self.other_photos_rgb_colors = np.zeros(shape=(len(self.input_images), color_size))

        # Get the average color of each image
        self.get_all_image_average_colors()

        # Using the distance between two points calculate
        self.indices_of_closest_image = self.calculate_distances()

        # Create the mosaic
        self.mosaic_image = self.create_image_grid()
        self.mosaic_image.save(output_file)
        print("Saved image successfully")

    def process_target_image(self, input_photo_path, enlargement):
        img = load_image(input_photo_path)
        img = resize_image_by_factor(img, enlargement)
        self.mosaic_image_size = img.size
        print(f"Upscale original image to {img.size}")
        return img

    def get_total_number_of_tiles(self):
        # Get the width and height of the image
        img_width, img_height = self.mosaic_image_size

        # Calculate the number of rows and columns in the grid
        cols = img_width // self.tile_image_size[0]
        rows = img_height // self.tile_image_size[1]
        return rows, cols

    def subdivide_original_image(self, initial_image):
        # Get the tile size into parameters to make code more readable
        tile_width, tile_height = self.tile_image_size

        # Mosaic number of tiles
        rows, cols = self.mosaic_image_shape

        # Iterate over the rows and columns of the grid
        row_cols_iterable = enumerate(itertools.product(range(rows), range(cols)))

        for k, (row, col) in tqdm(row_cols_iterable, desc='Dividing Original Image'):
            self.image_coordinates_in_order.append((row, col))
            top_left_point_coordinates = (col * tile_width, row * tile_height)
            bottom_right_point_coordinates = ((col + 1) * tile_width, (row + 1) * tile_height)
            coordinates = top_left_point_coordinates + bottom_right_point_coordinates
            cropped_img = initial_image.crop(coordinates)
            self.target_img_rgb_colors[k] = get_average_rgb_from_image(cropped_img)


    def process_input_images(self, images_directory, fit_method_name):
        img_list = self.get_images(images_directory)
        for image in tqdm(img_list, desc='Processing Input Images'):
            self.input_images.append(fit_image_to_tile(image, self.tile_image_size, fit_method_name))

    def get_images(self, images_directory):
        file_list = list(pathlib.Path(images_directory).glob('*[png|jpeg|jpg|PNG|JPEG|JPG]'))
        images = []
        for fp in file_list:
            img = load_image(fp)
            if img:
                images.append(img)
        return images

    def get_all_image_average_colors(self):
        iterable = enumerate(self.input_images)
        for i, img in tqdm(iterable, desc='Getting average color for the base image'):
            self.other_photos_rgb_colors[i] = get_average_rgb_from_image(img)

    def calculate_distances(self):
        A = self.target_img_rgb_colors
        B = self.other_photos_rgb_colors

        # Array A and B are 2D numpy arrays with shape (n, 3) and (m, 3), respectively
        # where n and m are the number of points in array A and array B, respectively
        # and each row represents a point with x, y, z coordinates

        # Calculate the pairwise differences between the x, y, and z coordinates
        differences = A[:, np.newaxis, :] - B[np.newaxis, :, :]

        # Calculate the pairwise Euclidean distances using the differences and the Euclidean distance formula
        distances = np.sqrt(np.sum(differences ** 2, axis=-1))

        # Find the indices of the minimum distances for each point in array A
        closest_points = np.argmin(distances, axis=1)

        return closest_points

    def oned_index_to_twod_index(self, i):
        return self.image_coordinates_in_order[i]

    def create_image_grid(self):
        tile_width, tile_height = self.tile_image_size

        grid_img = Image.new("RGB", self.mosaic_image_size)
        iterable = range(len(self.indices_of_closest_image))
        for i in tqdm(iterable, desc='Creating Photo Mosaic Original Image'):
            row, col = self.oned_index_to_twod_index(i)
            tile = self.input_images[self.indices_of_closest_image[i]]
            grid_img.paste(tile, (col * tile_width, row * tile_height))

        return grid_img


if __name__ == "__main__":
    m = Mosaic(input_photo_path="../../sample_photos/Figu SIMU Astrid.png",
               other_photos_path="../../sample_photos",
               enlargement=10,
               tile_size=(50, 50),
               output_file="output.png",
               fit_method="crop")
