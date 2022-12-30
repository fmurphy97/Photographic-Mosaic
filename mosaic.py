import itertools
import numpy as np
import pathlib
from tqdm import tqdm
from PIL import Image
import utilities


class Mosaic:
    PICTURE_MODE = 'RGB'

    def __init__(self, target_image_path, other_photos_path, tile_size, output_file, enlargement=1,
                 fit_method="resize", method_to_select_images="closest"):
        """
        Class that creates the whole mosaic
        :param target_image_path: where can the target image be found
        :param other_photos_path: the path to all the images that will be used to create the mosaic
        :param tile_size: the size of the tile in pixels
        :param output_file: path to the output mosaic
        :param enlargement: how many times will the image size be increased
        :param fit_method: the way in which this image will be resized, can be "resize" or "crop"
        """

        self.image_coordinates_in_order = []

        # Size of the tile image in pixels
        self.tile_image_size = tile_size

        # Read the target image, enlarge it, and
        target_image = self.process_target_image(target_image_path, enlargement)
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

        # Using the distance between two points calculate
        self.pairwise_distances = self.calculate_distances()
        self.indices_of_best_images = self.assign_best_image(selected_method=method_to_select_images)

        # Create the mosaic
        self.mosaic_image = self.create_image_grid()
        self.mosaic_image.save(output_file)
        print("Saved image successfully")

    def process_target_image(self, input_photo_path: str or pathlib.Path, enlargement: float) -> Image:
        """
        Loads image from path, resizes it, and stores it´s dimensions
        :param input_photo_path: where can the target image be found
        :param enlargement: how many times will the image size be increased
        :return: resized target image
        """
        img = utilities.load_image(filepath=input_photo_path, image_mode=Mosaic.PICTURE_MODE)
        img = utilities.resize_image_by_factor(img, enlargement)
        self.mosaic_image_size = img.size
        print(f"Upscale original image to {img.size}")
        return img

    def get_total_number_of_tiles(self) -> tuple[int, int]:
        """
        Using the tile size and the image size  calculates the total number of tiles that will be in the mosaic
        :return: the number of rows and columns of tiles that will be in the mosaic
        """
        # Get the width and height of the image
        img_width, img_height = self.mosaic_image_size

        # Calculate the number of rows and columns in the grid
        cols = img_width // self.tile_image_size[0]
        rows = img_height // self.tile_image_size[1]
        return rows, cols

    def subdivide_original_image(self, initial_image: Image):
        """
        Divides the target image into smaller pieces and calculates (and stores) the average color for each of them
        :param initial_image: target image
        """
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
            self.target_img_rgb_colors[k] = utilities.get_average_rgb_from_image(cropped_img)

    def process_input_images(self, images_directory: str or pathlib.Path, fit_method_name: str):
        """
        For each input image resizes it and then calculates the average color, and stores it
        :param images_directory: the path to all the images that will be used to create the mosaic
        :param fit_method_name: the way in which this image will be resized, can be "resize" or "crop"
        """
        img_list = utilities.find_images_in_path(images_directory)
        iterable = enumerate(img_list)

        for i, image in tqdm(iterable, desc='Processing Input Images'):
            # Resize the image
            resized_image = utilities.fit_image_to_tile(image, self.tile_image_size, fit_method_name)

            # Get the average color of the image
            self.other_photos_rgb_colors[i] = utilities.get_average_rgb_from_image(image)

            # Store it in the input images
            self.input_images.append(resized_image)

    def calculate_distances(self) -> np.array:
        """
        Calculates the distance between the average color in all the (small) images in the original image and the ones
        in the input image
        :return: the distance matrix between these two
        """
        A = self.target_img_rgb_colors
        B = self.other_photos_rgb_colors

        # Array A and B are 2D numpy arrays with shape (n, 3) and (m, 3), respectively
        # where n and m are the number of points in array A and array B, respectively
        # and each row represents a point with x, y, z coordinates

        # Calculate the pairwise differences between the x, y, and z coordinates
        differences = A[:, np.newaxis, :] - B[np.newaxis, :, :]

        # Calculate the pairwise Euclidean distances
        distances = np.sqrt(np.sum(differences ** 2, axis=-1))

        return distances

    def assign_best_image(self, selected_method="closest") -> np.array:
        """
        Using the selected method assigns which image will go in each tile space
        :param selected_method: the method that will be used to select the image. May be "closest" or "optimize_uses"
        :return: the indices (in the array B) of the selected images ordered by their assignation in A
        """

        # TODO: add this to an enum
        # TODO: add method that optimizes the number of times each image is used vs the closest image:
        #  min z = for every image: (img_distance / (sqrt(3) * 255)) - max(uses_uses_by_image) - 1), 0)
        methods_by_name = {"closest": self.assign_image_by_closest_point}
        method_to_use = methods_by_name[selected_method]
        return method_to_use()

    def assign_image_by_closest_point(self):
        """
        Selects the closest point (in B) for each image in A
        :return: the indices of the closest points
        """
        # Find the indices of the minimum distances for each point in array A
        indices_of_closest_points = np.argmin(self.pairwise_distances, axis=1)

        return indices_of_closest_points

    def one_dim_index_to_two_dim_index(self, i: int) -> tuple[int, int]:
        """
        Using an index in 1-D returns the corresponding index of the 2-D array
        :param i: the index
        :return: the index corresponding to the index in the 2-D array
        """
        return self.image_coordinates_in_order[i]

    def create_image_grid(self) -> Image:
        """
        Pastes each selected image into a final image
        :return: the final mosaic image
        """
        tile_width, tile_height = self.tile_image_size

        grid_img = Image.new("RGB", self.mosaic_image_size)
        iterable = range(len(self.indices_of_best_images))
        for i in tqdm(iterable, desc='Creating Photo Mosaic Original Image'):
            row, col = self.one_dim_index_to_two_dim_index(i)
            tile = self.input_images[self.indices_of_best_images[i]]
            grid_img.paste(tile, (col * tile_width, row * tile_height))

        return grid_img


if __name__ == "__main__":
    m = Mosaic(target_image_path="all_images/input.jpg",
               other_photos_path="all_images/input_photos",
               enlargement=10,
               tile_size=(50, 50),
               output_file="all_images/output.png",
               fit_method="crop")
