import cv2
import numpy as np
import random
import math
import sys
import matplotlib.pyplot as plt
import argparse
import glob
from tqdm import tqdm, trange

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img = gain_compensation(img)
    img = cylindrical_project(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def gain_compensation(images):
    
    # Calculate the average pixel intensity of each image
    base_max = 161.75303610596816
    exposure_max = 200.95138424123405
    # avg_intensities = [np.mean(img) for img in images]
    # max_intensity = max(avg_intensities)
    # print(max_intensity);exit()
    max_intensity = exposure_max
    # max_intensity = base_max
    
    # Normalize the intensities
    normalized_images = images * math.sqrt((np.mean(images)/max_intensity))
    
    return normalized_images.astype(np.uint8)

# pth_list = sorted(glob.glob('./Photos/{}/*'.format("Exposure")))
# images = []
# for path in pth_list:
#     images.append(cv2.imread(path))

# gain_compensation(images)


def cylindrical_project(img):
    """Inverse interpolation is applied to find the cylindrical projection of
    the original image.
    """
    cylindrical_img = np.zeros_like(img)
    height, width, depth = cylindrical_img.shape

    f = 1550 # focal length

    centerX = width / 2
    centerY = height / 2

    # pointX, pointY are coordinates in planar axis;
    # i, j, k are coordinates in cylindrical axis.
    for i in range(width):
        for j in range(height):
            theta = (i - centerX) / f
            pointX = int(f * np.tan((i - centerX) / f) + centerX)
            pointY = int((j - centerY) / np.cos(theta) + centerY)

            for k in range(depth):
                if 0 <= pointX < width and 0 <= pointY < height:
                    cylindrical_img[j, i, k] = img[pointY, pointX, k]
                else:
                    cylindrical_img[j, i, k] = 0

    return cylindrical_img
    


def parse_args():
    parser = argparse.ArgumentParser(description='argument settings')
    parser.add_argument('-t', '--testcase', help='type = "str", specify the test case', type=str, default="Base")
    parser.add_argument('-n', '--num', help='type = "int", number of the images to stitch', type=int, default=None)

    args = parser.parse_args()

    return args

class ImageStitching():
    def __init__(self, args):
        assert args.testcase in ['Base', 'Exposure'], "Testcase should be in ['Base', 'Exposure']"

        self.args = args
        self.testcase= args.testcase

        pth_list = sorted(glob.glob('./Photos/{}/*'.format(args.testcase)))
        
        if args.num is not None:
            assert args.num <= len(pth_list), "Number of images out of range, maximun = {}".format(len(self.img_pth_list[0]))
            self.img_pth_list = pth_list[:args.num]
        else:
            self.img_pth_list = pth_list

    def detect_featrues(self, img):
        sift = cv2.SIFT_create()
        kpts, features = sift.detectAndCompute(img, None)

        return kpts, features

    def match_features(self, kpts1, kpts2, feature1, feature2):
        matches, good_matches, good_matches_pos = [], [], []

        # find min-distance kpts pairs and second min-distance kpts pairs
        for i in tqdm(feature1):
            feat_arr = np.tile(i.reshape(1, 128), (feature2.shape[0], 1))
            dist_arr = np.linalg.norm(feat_arr - feature2, axis=1).reshape(-1)

            part_idx = np.argpartition(dist_arr, 1)
            matches.append([(kpts2[part_idx[0]], dist_arr[part_idx[0]]),
                            (kpts2[part_idx[1]], dist_arr[part_idx[1]])])

        # ratio test
        for k in range(len(matches)):
            d1 = matches[k][0][1]
            d2 = matches[k][1][1]

            if d1 < 0.75 * d2:
                good_matches.append([kpts1[k], matches[k][0][0]])

        # store kpts coordinates
        for (match1, match2) in good_matches:
            pos1 = (int(match1.pt[0]), int(match1.pt[1]))
            pos2 = (int(match2.pt[0]), int(match2.pt[1]))
            good_matches_pos.append([pos1, pos2])
        
        good_matches_pos = np.transpose(np.array(good_matches_pos), (1, 0, 2))

        return good_matches_pos

    def solve_homography(self, src, dst):
        A = []
        for r in range(len(src)):
            A.append([-src[r, 0], -src[r, 1], -1, 0, 0, 0, src[r, 0] * dst[r, 0], src[r, 1] * dst[r, 0], dst[r, 0]])
            A.append([0, 0, 0, -src[r, 0], -src[r, 1], -1, src[r, 0] * dst[r, 1], src[r, 1] * dst[r, 1], dst[r, 1]])

        # Solve s ystem of linear equations Ah = 0 using SVD
        u, s, vt = np.linalg.svd(A)
        
        # pick H from last line of vt
        H = vt[8].reshape(3, 3)

        # normalization, let H[2,2] equals to 1
        H /= H[2, 2]

        return H

    def RANSAC(self, matches_pos):
        THRESHOLD = 5.0
        NUM_ITERATIONS = 8000
        NUM_RANDOM_SAMPLES = 4

        num_samples = matches_pos.shape[1]
        max_inliners = 0
        best_H = None
        
        for _ in trange(NUM_ITERATIONS):
            # sample kpts for computing RANSAC
            sample_idx = random.sample(range(num_samples), NUM_RANDOM_SAMPLES)
            mask_validate = np.ones(num_samples, dtype=bool)
            mask_validate[sample_idx] = False

            # mask out sample kpts for validate inliners
            pos1_validate = matches_pos[0, mask_validate, :].copy()
            pos2_validate = matches_pos[1, mask_validate, :].copy()
            pos1_validate = np.hstack([pos1_validate, np.ones((num_samples - NUM_RANDOM_SAMPLES, 1))]).T
            pos2_validate = np.hstack([pos2_validate, np.ones((num_samples - NUM_RANDOM_SAMPLES, 1))]).T

            # solve homography based on sampled kpts
            cur_H = self.solve_homography(matches_pos[0, sample_idx], matches_pos[1, sample_idx])

            # compute inliners
            pos2_estimate = cur_H @ pos1_validate
            nonzero_idx = (pos2_estimate[2] > 0) # avoid divide zero number

            pos2_estimate = pos2_estimate[:, nonzero_idx]
            pos2_estimate = (pos2_estimate / np.tile(pos2_estimate[2], (3, 1))).T

            pos2_validate = pos2_validate[:, nonzero_idx].T

            dist = np.linalg.norm((pos2_validate[:, :2] - pos2_estimate[:, :2]), axis=1)
            num_inliners = np.sum(dist < THRESHOLD)

            # update H if more inliners founded
            if num_inliners > max_inliners:
                max_inliners = num_inliners
                best_H = cur_H

        return best_H

    def stitch_images(self, img1, img2, H):
        # find corners of new images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(4, 1, 2)
        corners2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(4, 1, 2)
        corners_transform = np.concatenate((cv2.perspectiveTransform(corners1, H), corners2), axis=0)
        
        xmin, ymin = np.min(corners_transform - 0.5, axis=0).reshape(-1).astype(np.int32)
        xmax, ymax = np.max(corners_transform + 0.5, axis=0).reshape(-1).astype(np.int32)

        # create affine translate matrix
        affine_translation = np.array([[1, 0, -xmin],
                                       [0, 1, -ymin],
                                       [0, 0, 1]], dtype=np.float64)
        transform_mat = affine_translation @ H

        # stitch two images
        new_img_size = (xmax - xmin, ymax - ymin)

        # warp image 1: rotation (rotate to image 2 coordinates) + translation (shift to origin)
        warp_img1 = cv2.warpPerspective(src=img1, M=transform_mat, dsize=new_img_size)

        # shift image 2: translation (shift the same amount as image 1)
        warp_img2 = cv2.warpPerspective(src=img2, M=affine_translation, dsize=new_img_size)

        # blending in intersection
        intersection_mask = (warp_img1 > 0) & (warp_img2 > 0)


        stitched_img = warp_img1 + warp_img2
        m1 = cv2.mean(warp_img1[intersection_mask])[0]
        m2 = cv2.mean(warp_img2[intersection_mask])[0]
        r1 = m1/(m1+m2)
        r2 = m2/(m1+m2)

        stitched_img[intersection_mask] = cv2.addWeighted(warp_img1[intersection_mask], r1,
                                                          warp_img2[intersection_mask], r2, 3).reshape(-1)

        return stitched_img


    def run(self):
        print('Testcase: {}, stitching from m1 ~ m{}'.format(self.testcase, len(self.img_pth_list)))

        img1, img1_gray = read_img(self.img_pth_list[0])
        for i in range(1, len(self.img_pth_list)):
            img2, img2_gray = read_img(self.img_pth_list[i])

            print()
            print('-' * 20 + ' Stitching with m{} '.format(i + 1) + '-' * 20)
            print('Step1: Detecting keypoints (feature) on the images')
            kpts1, feature1 = self.detect_featrues(img1_gray)
            kpts2, feature2 = self.detect_featrues(img2_gray)

            print('Step2: Finding features correspondences (feature matching)')
            good_matches = self.match_features(kpts1, kpts2, feature1, feature2)

            print('Step3: Computing homography matrix')
            best_H = self.RANSAC(good_matches)

            print('Step4: Stitch images')
            result = self.stitch_images(img1, img2, best_H)
            
            img1, img1_gray = result, img_to_gray(result)

        cv2.imwrite("{}_m1-m{}.jpg".format(self.testcase, len(self.img_pth_list)), img1)
        creat_im_window("{}_m1-m{}".format(self.testcase, len(self.img_pth_list)), img1)
        im_show()

if __name__ == '__main__':
    args = parse_args()

    stitcher = ImageStitching(args)
    stitcher.run()










####### FAILED ##########
def compute_mean_intensity(self, image, overlap_region):
        return np.mean(image[overlap_region])

def compute_global_error(self, images, overlaps, gains, sigma_N=10.0, sigma_g=0.1):
    """
    Compute the global error function for gain computation.
    
    Parameters:
        images (list of ndarrays): List of images.
        overlaps (list of ndarrays): List of boolean masks indicating overlap regions.
        gains (list of floats): List of current gains for each image.
        sigma_N (float): Standard deviation of the normalized intensity error.
        sigma_g (float): Standard deviation of the gain.
    
    Returns:
        float: Global error.
    """
    num_images = len(images)
    global_error = 0.0
    
    for i in range(num_images):
            for j in range(num_images):
                if i != j:
                    overlap_mask = overlaps[i][j]
                    N_ij = np.count_nonzero(overlap_mask)
                    mean_intensity_ij = compute_mean_intensity(images[i], overlap_mask)
                    mean_intensity_ji = compute_mean_intensity(images[j], overlap_mask)
                    
                    term1 = (gains[i] * mean_intensity_ij - gains[j] * mean_intensity_ji) ** 2 / (sigma_N ** 2 * N_ij)
                    term2 = (1 - gains[i]) ** 2 / sigma_g ** 2
                    
                    global_error += term1 + term2
    
    return 0.5 * global_error

def compute_gain(self, images, overlaps, initial_gains, sigma_N=10.0, sigma_g=0.1, max_iter=100, tol=1e-4):
    """
    Compute the gains for each image using the global error minimization.
    
    Parameters:
        images (list of ndarrays): List of images.
        overlaps (list of ndarrays): List of boolean masks indicating overlap regions.
        initial_gains (list of floats): Initial gains for each image.
        sigma_N (float): Standard deviation of the normalized intensity error.
        sigma_g (float): Standard deviation of the gain.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
    
    Returns:
        list of floats: Computed gains for each image.
    """
    num_images = len(images)
    gains = initial_gains.copy()
    prev_error = float('inf')
    
    for _ in range(max_iter):
        global_error = compute_global_error(images, overlaps, gains, sigma_N, sigma_g)
        
        if abs(global_error - prev_error) < tol:
            break
        
        prev_error = global_error
        
        for j in range(num_images):
            term1 = 0.0
            term2 = 0.0
            
            for i in range(num_images):
                if i != j:
                    overlap_mask = overlaps[i][j]
                    N_ij = np.count_nonzero(overlap_mask)
                    mean_intensity_ij = compute_mean_intensity(images[i], overlap_mask)
                    mean_intensity_ji = compute_mean_intensity(images[j], overlap_mask)
                    
                    term1 += N_ij * mean_intensity_ij ** 2
                    term2 += N_ij * mean_intensity_ij * mean_intensity_ji * gains[i]
            
            gains[j] = term2 / (sigma_N ** 2 * term1 + sigma_g ** 2)
    
    return gains