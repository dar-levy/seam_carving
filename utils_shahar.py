import math

import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod

def find_nth_smallest_proper_way(a, n):
    return np.partition(a, n-1)[n-1]
class SeamImage:
    def __init__(self, img_path, vis_seams=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            method (str) (a or b): a for Hard Vertical and b for the known Seam Carving algorithm
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path

        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T

        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()

        self.h, self.w = self.rgb.shape[:2]

        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices 
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))

    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        gs_image = np.dot(self.rgb, self.gs_weights)
        gs_image = np.pad(gs_image, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0.5)
        return gs_image

    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            In order to calculate a gradient of a pixel, only its neighborhood is required.
        """
        gradient_magnitude_matrix = np.copy(self.gs)
        for row in range(0, len(gradient_magnitude_matrix), 1):
            for col in range(0, len(gradient_magnitude_matrix[row]), 1):
                ev = self.calc_energy_vertical(gradient_magnitude_matrix, row, col)
                eh = self.calc_energy_vertical(gradient_magnitude_matrix, row, col)

                gradient_magnitude_matrix[row][col] = np.sqrt(ev ** 2 + eh ** 2)

        return gradient_magnitude_matrix

    def calc_energy_vertical(self, mat, row, col):
        if row > len(mat) - 1:
            row = 0
        if col > len(mat[row]) - 2:
            col = 0
        ev_col = col + 1 if col < len(mat[row]) - 1 else col - 1
        ev = np.abs(mat[row][col] - mat[row][ev_col])
        return ev

    def calc_energy_horizontal(self, mat, row, col):
        if row > len(mat) - 2:
            row = 0
        if col > len(mat[row]) - 1:
            col = 0
        eh_row = row + 1 if row < len(mat) - 1 else row - 1
        eh = np.abs(mat[row][col] - mat[eh_row][col])
        return eh

    def calc_M(self):
        pass

    def seams_removal(self, num_remove):
        pass

    def seams_removal_horizontal(self, num_remove):
        pass

    def seams_removal_vertical(self, num_remove):
        pass

    def rotate_mats(self, clockwise):
        pass

    def init_mats(self):
        pass

    def update_ref_mat(self):
        pass

    def backtrack_seam(self):
        pass

    def remove_seam(self):
        pass

    def reinit(self):
        """ re-initiates instance
        """
        self.__init__(self.path)

    @staticmethod
    def load_image(img_path):
        return np.asarray(Image.open(img_path)).astype('float32') / 255.0


class VerticalSeamImage(SeamImage):
    def __init__(self, *args, **kwargs):
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.tmp_mask = np.copy(self.cumm_mask)
            self.seams_rgb = np.pad(self.rgb, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0.5)
            self.resized_rgb = np.pad(self.resized_rgb, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0.5)
            self.M = self.calc_M()
            self.mt = np.ndarray(shape=(self.M.shape[0],
                                              self.M.shape[1],
                                              self.M.shape[2] + 1), dtype=np.int32)
            self.mt = VerticalSeamImage.calc_bt_mat(self.M, self.mt)

        except NotImplementedError as e:
            print(e)

    def calc_M(self):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        M = np.copy(self.E)
        for row in range(1, len(M) - 1):
            for col in range(0, len(M[row]) - 1):
                cl = np.abs(self.gs[row][col + 1] - self.gs[row][col - 1]) + \
                     np.abs(self.gs[row - 1][col] - self.gs[row][col - 1])

                cv = np.abs(self.gs[row][col + 1] - self.gs[row][col - 1])

                cr = np.abs(self.gs[row][col + 1] - self.gs[row - 1][col]) + \
                     np.abs(self.gs[row][col + 1] - self.gs[row][col - 1])

                M[row][col] += min(M[row - 1][col - 1] + cl,
                                   M[row - 1][col] + cv,
                                   M[row - 1][col + 1] + cr)

        return M

    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams

        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to supprt:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """
        self.seams_removal_vertical(num_remove)
        # raise NotImplementedError("TODO: Implement SeamImage.seams_removal")

    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        self.gs = np.transpose(self.gs, (1, 0, 2))
        self.E = np.transpose(self.E, (1, 0, 2))
        self.cumm_mask = np.transpose(self.cumm_mask, (1, 0, 2))
        self.seams_rgb = np.transpose(self.seams_rgb, (1, 0, 2))
        self.resized_rgb = np.transpose(self.resized_rgb, (1, 0, 2))
        self.mt = np.ndarray(shape=(self.E.shape[0],
                                    self.E.shape[1],
                                    self.E.shape[2] + 1), dtype=np.int32)
        self.mt = VerticalSeamImage.calc_bt_mat(self.M, self.mt)

        self.seams_removal(num_remove)

        self.gs = np.transpose(self.gs, (1, 0, 2))
        self.E = np.transpose(self.E, (1, 0, 2))
        self.cumm_mask = np.transpose(self.cumm_mask, (1, 0, 2))
        self.seams_rgb = np.transpose(self.seams_rgb, (1, 0, 2))
        self.resized_rgb = np.transpose(self.resized_rgb, (1, 0, 2))
        self.mt = np.ndarray(shape=(self.M.shape[0],
                                    self.M.shape[1],
                                    self.M.shape[2] + 1), dtype=np.int32)
        self.mt = VerticalSeamImage.calc_bt_mat(self.M, self.mt)


    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        cols_indices_to_remove = list()
        tmp_mask = np.ones_like(self.gs, dtype=bool)
        for _ in range(num_remove):
            self.E = self.calc_gradient_magnitude()
            self.M = self.calc_M()

            seam = self.backtrack_seam()
            for row, col in seam:
                self.cumm_mask[row][col] = False
                tmp_mask[row][col] = False

            t = np.roll(self.gs, 1, axis=0)
            self.gs = np.where(tmp_mask, self.gs, t)
            self.gs = np.delete(self.gs, -1, axis=1)

            t = np.roll(self.E, 1, axis=0)
            self.E = np.where(tmp_mask, self.E, t)
            self.E = np.delete(self.E, -1, axis=1)

            t = np.roll(self.M, 1, axis=0)
            self.M = np.where(tmp_mask, self.M, t)
            self.M = np.delete(self.M, -1, axis=1)

            t = np.roll(self.resized_rgb, 1, axis=0)
            self.resized_rgb = np.where(tmp_mask, self.resized_rgb, t)
            self.resized_rgb = np.delete(self.resized_rgb, -1, axis=1)

            t = np.roll(tmp_mask, 1, axis=0)
            tmp_mask = np.where(tmp_mask, tmp_mask, t)
            tmp_mask = np.delete(tmp_mask, -1, axis=1)


        self.seams_rgb = np.where(self.cumm_mask, self.seams_rgb, (1, 0, 0))


    def backtrack_seam(self):
        """ Backtracks a seam for Seam Carving as taught in lecture
        """
        last_row = len(self.M) - 2
        col = np.argmin(self.M[last_row][1:-1])
        seam = []
        for row in range(last_row, 1, -1):
            seam.append([row, col])

            cl = np.abs(self.gs[row][col + 1] - self.gs[row][col - 1]) + \
                 np.abs(self.gs[row - 1][col] - self.gs[row][col - 1])

            cv = np.abs(self.gs[row][col + 1] - self.gs[row][col - 1])

            if self.M[row][col] == self.E[row][col] + self.M[row - 1][col] + cv:
                continue

            elif self.M[row][col] == self.E[row][col] + self.M[row - 1][col - 1] + cl:
                col -= 1

            elif col < self.M.shape[1] - 3:
                col += 1
        return seam

    def remove_seam(self):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """
        raise NotImplementedError("TODO: Implement SeamImage.remove_seam")

    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seamn to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO: Implement SeamImage.seams_addition")

    def seams_addition_horizontal(self, num_add):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    def seams_addition_vertical(self, num_add):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_vertical")

    @staticmethod
    @jit(nopython=True)
    def calc_bt_mat(M, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.

        Recommnded parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a rederence type. changing it here may affected outsde.
        """
        # backtrack_mat.reshape(backtrack_mat.shape[0], backtrack_mat.shape[1], backtrack_mat.shape[2] + 1)

        for row in range(len(backtrack_mat)):
            for col in range(len(backtrack_mat[row])):
                backtrack_mat[row][col] = [row, col]

        return backtrack_mat


def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    scale_factors_new = []
    orig_y = orig_shape[0]
    orig_x = orig_shape[1]

    return [int(scale_factors[0] * orig_y), int(scale_factors[1] * orig_shape[1])]
    # raise NotImplementedError("TODO: Implement SeamImage.scale_to_shape")


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """

    orig_shape = shapes[0]
    new_shape = shapes[1]
    seam_img.seams_removal_horizontal(orig_shape[0] - new_shape[0])
    seam_img.seams_removal_vertical(orig_shape[1] - new_shape[1])
    return seam_img.resized_rgb
    # raise NotImplementedError("TODO: Implement SeamImage.resize_seam_carving")


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)

    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org

    scaled_x_grid = [get_scaled_param(x, in_width, out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y, in_height, out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid, dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid, dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:, x1s] * dx + (1 - dx) * image[y1s][:, x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:, x1s] * dx + (1 - dx) * image[y2s][:, x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image
