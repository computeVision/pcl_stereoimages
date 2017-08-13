#!/usr/bin/env python3

import numpy as np
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import os
from glob import glob
import time
import pickle
import random
import config
import cv2

# I added it.
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

glob_counter = 0

# todo delete before submission
print('opencv version ' + cv2.__version__)
import pdb

def save_plt(path, remove_axis=False):
  """ Saves the current figure. """
  plt.tight_layout()
  if remove_axis:
    plt.axis('off')
  plt.savefig(path, dpi=300)


def kps2np(kps):
  kps_np = np.empty((len(kps), 2), dtype=np.float32)
  for row, kp in enumerate(kps):
    kps_np[row, 0] = kp.pt[0]
    kps_np[row, 1] = kp.pt[1]
  return kps_np


def dmatch2np(matches):
  matches_np = np.empty((len(matches), 2), dtype=np.int32)
  for row, match in enumerate(matches):
    matches_np[row, 0] = match.queryIdx
    matches_np[row, 1] = match.trainIdx
  return matches_np

def appendimages(img1, img2):
  """ return a new image that appends the two images side-by-side."""
  # copied and adapted from there: https://github.com/shackenberg/Minimal-Bag-of-Visual-Words-Image-Classifier/blob/master/sift.py
  print('-------- appendimages --------')
  # select the image with the fewest rows and fill in enough empty rows
  rows1 = img1.shape[0]
  rows2 = img2.shape[0]

  img1 = np.dstack([img1, img1, img1])  # to colorize it to gray
  img2 = np.dstack([img2, img2, img2])

  if rows1 < rows2:
    img1 = np.concatenate((img1, np.zeros((rows2 - rows1, img1.shape[1], 3), dtype='uint8')), axis=0)
  else:
    img2 = np.concatenate((img2, np.zeros((rows1 - rows2, img2.shape[1], 3), dtype='uint8')), axis=0)

  return np.concatenate((img1, img2), axis=1)


def draw_matches(img1, kp1, img2, kp2, matches):
  '''
  Getting numpy arrays for images and keypoints and a list of matches with numpy objects
  :param img1: left image
  :param kp1: np.array
  :param img2: right image
  :param kp2: np.array
  :param matches: list(np.array)
  :return: two images are sticked together to their side
  '''

  cols1 = img1.shape[1]
  appended_image = appendimages(img1, img2)

  for mat in matches:
    x1, y1 = kp1[mat.queryIdx]
    x2, y2 = kp2[mat.trainIdx]

    x1 = np.int(x1)
    x2 = np.int(x2)
    y1 = np.int(y1)
    y2 = np.int(y2)

    color = random.sample(range(256), 3)
    cv2.line(appended_image, (x1, y1), (x2 + cols1, y2), color, thickness=5)
    cv2.circle(appended_image, (x1, y1), 8, (255, 0, 0), thickness=1)  # left circle on line
    cv2.circle(appended_image, (x2 + cols1, y2), 8, (255, 0, 0), thickness=1)  # right circle on line

  plt.figure(300)
  plt.imshow(appended_image)
  plt.show()

def plt_matches(left_im, right_im, left_kp, right_kp, matches):
  # TODO implement the visualization of keypoint matches
  print(' --------------- plt matches ---------------')
  matches = sorted(matches, key=lambda val: val.distance)
  im3 = appendimages(left_im, right_im)
  im = cv2.drawMatches(left_im, left_kp, right_im, right_kp, im3, matches)

  plt.figure(200)
  plt.imshow(im)
  plt.show()


def draw_kp(img1, kp1, img2, kp2):
  cols1 = img1.shape[1]
  appended_image = appendimages(img1, img2)

  for iter_kp in range(kp1.shape[0]):
    x1, y1 = kp1[iter_kp]
    x2, y2 = kp2[iter_kp]
    x1 = np.int(x1)
    x2 = np.int(x2)
    y1 = np.int(y1)
    y2 = np.int(y2)

    color = random.sample(range(256), 3)
    cv2.line(appended_image, (x1, y1), (x2 + cols1, y2), color, thickness=5)
    cv2.circle(appended_image, (x1, y1), 8, (255, 0, 0), thickness=2)  # left circle on line
    cv2.circle(appended_image, (x2 + cols1, y2), 8, (255, 0, 0), thickness=2)  # right circle on line

  plt.figure()
  plt.imshow(appended_image)
  plt.show()

def checkout_contrast2(im_left, im_right):
  '''
  Get image left and right and change the values of the contrast
  :param im_left: left handside image
  :param im_right: corresponding right handside image
  :return: current contrast value
  '''
  # todo use this one and not the oder contast function
  cur_left = 0
  cur_right = 0
  cur_con = 0

  for contrast in (0.0001, 0.001, 0.01):
    print('start comparison with contrast ', contrast)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=20000, contrastThreshold=contrast)
    kp_left, des_left = sift.detectAndCompute(im_left, None)
    kp_right, des_right = sift.detectAndCompute(im_right, None)

    # to not delete
    # plt.figure(300)
    # plt.imshow(im_left)
    # plt.figure(301)
    # plt.imshow(im_right)
    # plt.show()

    lengthkp_l = len(kp_left)
    lengthkp_r = len(kp_right)

    if lengthkp_l > cur_left or lengthkp_r > cur_right:
      cur_con = contrast
      cur_left = lengthkp_l
      cur_right = lengthkp_r

  print('--------- check best contrast for most feature points ------------')
  print("current contrast: ", cur_con)
  print("current kp_left : ", cur_left)
  print("current kp_right: ", cur_right)
  print('------------------------------------------------------------------')

  return cur_con


def rigid_motion(P, Q):
  # TODO compute the rigid motion R,t between the two point clouds

  P_center = np.mean(P, axis=1).reshape((-1, 1))
  Q_center = np.mean(Q, axis=1).reshape((-1, 1))

  cov = np.dot((Q - Q_center), ((P - P_center).T))

  U, S, Vt = np.linalg.svd(cov)

  R = np.dot(Vt, U)
  T = P_center - R.dot(Q_center)

  return R, T

def ransac(X: object, Y: object, matches, kp0, kp1, n: object, iters: object, min_dist: object, img1: object,
           img2: object) -> object:
  '''
  # https://en.wikipedia.org/wiki/Random_sample_consensus
  # -----------------------------------------------------
  # 6 samples: 3rotation und 3translation
  # jede iteration is unabhaengig von der vorherigen.
  # merken das beste set
  # am ende nochmal rotation und translation mit diesen inliers berechnen.
  :param X: a set of observed data points
  :param Y: a model that can be fitted to data points
  :param n: the minimum number of data values required to fit the model. 3 for transl and 3 for rotation
  :param iters: the maximum number of iterations allowed in the algorithm
  :param min_dist: a threshold value for determining when a data point fits a model, is in milimeter
  :return: bestfit – model parameters which best fit the data (or nul if no good model is found)
  '''
  # TODO compute the rotation matrix R and and translation matrix using a RANSAC loop
  print('---- ransac ----')

  # def rigid_motion2(X, Y):
  #   # TODO compute the rigid motion R,t between the two point clouds
  #
  #   X_mean = np.atleast_2d(np.mean(X, axis=1)).T  # make a np array from the list
  #   Y_mean = np.atleast_2d(np.mean(Y, axis=1)).T
  #
  #   u, s, v = np.linalg.svd((Y - Y_mean).dot((X - X_mean).T))  # compute U, V
  #   R = v.T.dot(u.T)
  #   t = X_mean - R.dot(Y_mean)
  #
  #   return R, t
  #
  # def random_pointclouds(x):
  #   return 2500 * np.random.random((3, len(x)))
  #
  # X1, X2, X3 = X = random_pointclouds(np.zeros(500))  # todo delete after depthmap is fixed
  # Y1, Y2, Y3 = Y = random_pointclouds(np.zeros(500))
  #
  # fig = plt.figure()
  # ax = Axes3D(fig)
  # ax.scatter(X1, X2, X3, color='r')
  # ax.scatter(Y1, Y2, Y3, color='g')
  # plt.show()
  #
  # rot = np.zeros((3, 3))
  #
  # rot[0, 1] = -1
  # rot[1, 0] = 1
  # rot[2, 2] = 1
  #
  # print(rot)
  # print('inv', np.linalg.inv(rot))
  #
  # X_rot = np.zeros_like(X)
  #
  # print(len(X))
  # X_rot = rot.dot(X)
  #
  # R, t = rigid_motion(X, X_rot)
  #
  # print('R', R)
  # print('Rt', R.T)
  # print('t', t)
  #
  # R2, t2 = rigid_motion2(X, X_rot)
  #
  # print('R2', R2)
  # print('t2', t2)
  #
  # pdb.set_trace()
  # #
  # print(len(X))

  inliers = []
  for _ in range(iters):
    idx = random.sample(range(len(matches)), n)
    X_rans = X[:, idx]
    Y_rans = Y[:, idx]

    R, t = rigid_motion(X_rans, Y_rans)

    pcl_transf = R.dot(Y) + t.reshape((-1, 1))

    dist = np.linalg.norm(pcl_transf - X, axis=0)
    curr_inls = np.where(dist < min_dist)[0]

    if len(inliers) < len(curr_inls):
      inliers = curr_inls

  print('inliers', len(inliers))
  X_inliers = np.array(X[:, inliers])
  Y_inliers = np.array(Y[:, inliers])

  # to not delete
  # draw_kp(img1, kp0[inliers, :], img2, kp1[inliers, :])

  R, t = rigid_motion(X_inliers, Y_inliers)
  return R, t, [X_inliers, Y_inliers]


def main():
  data_paths = sorted( glob(os.path.join('../data', '*.npz')) )
  sift = cv2.xfeatures2d.SIFT_create(nfeatures=20000, contrastThreshold=0.0001)
  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

  with open('../data/calib.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    K = data['K']
    T = data['T']
  focal_length = K[0, 0]

  # compute keypoints and features
  for idx, data_path in enumerate(data_paths):
    features_path = os.path.join('../data', 'features%04d.pkl' % idx)
    if not os.path.exists(features_path):
      data = np.load(data_path)
      im_left = data['gray_left']
      im_right = data['gray_right']

      # to not delete!!
      # cur_contrast = checkout_contrast(im_left, im_right)
      # checkout_contrast2(im_left, im_right)
      # continue

      print('compute descriptors %d/%d' % (idx+1, len(data_paths)))
      tic = time.time()

      mask_left = np.ones_like(im_left)
      mask_right = np.ones_like(im_right)
      highlights_left = np.asarray(np.where(im_left >= 250), dtype=np.int)
      highlights_right = np.asarray(np.where(im_left >= 250), dtype=np.int)

      win_size = 5
      for coord in highlights_left:
        mask_left[coord[0] - win_size:coord[0] + win_size, coord[1] - win_size:coord[1] + win_size] = 0
      for coord in highlights_right:
        mask_right[coord[0] - win_size:coord[0] + win_size, coord[1] - win_size:coord[1] + win_size] = 0

      # TODO compute SIFT image keypoints and descriptors for left and right
      # http://stamfordresearch.com/basic-sift-in-python/
      print('--------------- Compute descriptors and keypoints --------------------')
      kp_left, des_left = sift.detectAndCompute(im_left, mask_left)
      kp_right, des_right = sift.detectAndCompute(im_right, mask_right)

      # tmp = kps2np(kp_left)
      # plt.figure()
      # plt.imshow(im_left, cmap='gray')
      # plt.plot(tmp[:,0], tmp[:,1], 'r.')
      # plt.show()

      print('  took %f[s]' % (time.time() - tic))
      matches = bf.match(des_left, des_right)
      # draw_matches(im_left, kps2np(kp_left), im_right, kps2np(kp_right), matches)

      with open(features_path, 'wb') as f:
        pickle.dump(
          {'kp_right': kps2np(kp_right), 'kp_left': kps2np(kp_left), 'des_right': des_right, 'des_left': des_left}, f)

  print("parse stereo feature matching and triangulation")
  # sparse stereo feature matching and triangulation
  features_paths = sorted( glob( os.path.join('../data/', 'features*pkl') ) )
  for idx in range(len(features_paths)):
    stereo_matches_path = os.path.join('../data', 'stereo_matches%04d.pkl' % idx)
    if not os.path.exists(stereo_matches_path):
      with open(features_paths[idx], 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        kp_left = data['kp_left']
        kp_right = data['kp_right']
        des_left = data['des_left']
        des_right = data['des_right']

        data_img = np.load(data_path)
        im_left = data_img['gray_left']
        im_right = data_img['gray_right']

        print(' -------- match stereo  ------')
        tic = time.time()
        # TODO compute the descriptor matches using cross checking
        matches = bf.match(des_left, des_right)
        np_matches = dmatch2np(matches)
        # binary & solution found on: https://stackoverflow.com/questions/44413234/how-to-filter-a-numpy-array-based-on-two-conditions-one-depending-on-the-other
        ind = np.where((np.abs(kp_left[np_matches[:, 0], 1] - kp_right[np_matches[:, 1], 1]) < 4) &
                       (kp_left[np_matches[:, 0], 0] - kp_right[np_matches[:, 1], 0] >= 0))[0]
        goodmatches = np.array(matches)[ind]

        print('good ', len(goodmatches))
        # do not delete!
        # draw_matches(im_left, kp_left, im_right, kp_right, goodmatches)

        # TODO compute the sparse 3D point cloud xyz using the matches
        baseline = np.linalg.norm(T)
        K_inv = np.linalg.inv(K)
        xyz = np.zeros((3, len(goodmatches)))

        index_left = []
        for idx, match in enumerate(goodmatches):
          diff_horiz = kp_left[match.queryIdx, 0] - kp_right[match.trainIdx, 0]
          hom_vec = np.array([kp_left[match.queryIdx, 0], kp_left[match.queryIdx, 1], 1])
          # depth: z = focal_length*baseline / (x_l - x_r)
          z = focal_length * baseline / diff_horiz
          # viewing ray
          xyz[:, idx] = np.dot(K_inv, hom_vec) * z
          index_left.append(match.queryIdx)


        print('  took %f[s]' % (time.time() - tic))

        with open(stereo_matches_path, 'wb') as f:
          pickle.dump(
            {'kp': kp_left[np.asarray(index_left), :], 'des': des_left[np.asarray(index_left), :], 'xyz': xyz}, f)

  print("------------ forward matching ------------")
  # forward matching
  stereo_matches_paths = sorted(glob( os.path.join('../data/', 'stereo_matches*pkl') ))
  for idx in range(len(features_paths) - 1):
    fwd_matches_path = os.path.join('../data', 'fwd_matches%04d.pkl' % idx)
    if not os.path.exists(fwd_matches_path):
      with open(stereo_matches_paths[idx], 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        kp0 = data['kp']
        des0 = data['des']
      with open(stereo_matches_paths[idx + 1], 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        kp1 = data['kp']
        des1 = data['des']
        data_img = np.load(data_path)
        im_left = data_img['gray_left']
        im_right = data_img['gray_right']

        print('------------- match descriptors fwd -------------')
        tic = time.time()
        # TODO match the descriptors of two consecutive images usig a ratio test
        # we copied the code from there: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des0, des1, k=2)
        # Apply ratio test
        fwd_mat = []
        for m, n in matches:
          if m.distance < 0.75 * n.distance:  # 0.75 should be as it be. according to tutor.
            fwd_mat.append(m)
        print('  took %f[s]' % (time.time() - tic))
        # do not delete!
        draw_matches(im_left, kp0, im_right, kp1, fwd_mat)

        with open(fwd_matches_path, 'wb') as f:
          pickle.dump({'matches': dmatch2np(fwd_mat)}, f)

  R_cumulative = np.eye(3, 3)  # Matrices to transform the frames into the first one
  T_cumulative = np.zeros(
    (3, 1))  # They are multiplicated and added continuously to have only one transformation per frame in the end.

  fwd_matches_paths = sorted(glob( os.path.join('../data/', 'fwd_matches*pkl') ))
  for idx in range(len(features_paths) - 1):
    trans_path = os.path.join('../data', 'trans%04d.pkl' % idx)
    if not os.path.exists(trans_path):
      with open(stereo_matches_paths[idx], 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        kp0 = data['kp']
        des0 = data['des']
        xyz0 = data['xyz']
      with open(stereo_matches_paths[idx + 1], 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        kp1 = data['kp']
        des1 = data['des']
        xyz1 = data['xyz']
      with open(fwd_matches_paths[idx], 'rb') as f:
        matches = pickle.load(f, encoding='latin1')['matches']

      data_img = np.load(data_path)
      img1 = data_img['gray_left']
      img2 = data_img['gray_right']

      # TODO: compute R and t using the least-squares solution in a RANSAC loop
      R, T, inliers = ransac(xyz0[:, matches[:, 0]], xyz1[:, matches[:, 1]], matches, kp0[matches[:, 0], :],
                             kp1[matches[:, 1], :], n=6, iters=6000, min_dist=120, img1=img1, img2=img2)

      R_cumulative = R.dot(R_cumulative)  # multiply the R matrix for later transformation
      T_cumulative = R.dot(T_cumulative) + T  # add T for later transformation

      with open(trans_path, 'wb') as f:
        pickle.dump({'R': R_cumulative, 'T': T_cumulative, 'inliers': inliers}, f)

        # transform points & write ply file
  trans_paths = sorted(glob(os.path.join('../data/', 'trans*pkl')))

  for idx, data_path in enumerate(data_paths):
    print('write ply file %d/%d' % (idx + 1, len(data_paths)))

    data = np.load(data_path)
    roi = config.roi_left_xywh
    rgb = data['color_left'][roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
    xyz = data['depth']

    if idx == 0:
      xyz_trans = xyz[xyz[:, :, 2] > 0]
      rgb_trans = rgb[xyz[:, :, 2] > 0]

    else:
      with open(trans_paths[idx - 1], 'rb') as f:
        data = pickle.load(f)
        Rn = data['R']
        Tn = data['T'].flatten()

      xyz_trans = xyz[xyz[:, :, 2] > 0]
      rgb_trans = rgb[xyz[:, :, 2] > 0]

      xyz_trans = np.dot(xyz_trans, Rn.T) + Tn  # transform them into the first image

    jumps = 5
    idxs = np.arange(0, xyz_trans.shape[0], jumps)
    f = open("img_" + str(idx) + ".ply", 'w')
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex %s\n' % idxs.shape[0])
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    f.write('end_header\n')

    for (x, y, z), (r, g, b) in zip(xyz_trans[idxs], rgb_trans[idxs]):
      f.write("%f %f %f %i %i %i\n" % (x, y, z, r, g, b))

    f.close()

if __name__ == "__main__":
  main()
