""" Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
"""

import os
import sys
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "mayavi"))
import kitti_util as utils
import argparse

care_types = ['Car', 'Pedestrian', 'Cyclist']


class kitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split='trainval', pred_dir=None):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split = split

        if 'train' in self.split or 'val' in self.split:
            self.subset = 'training'
        elif 'test' in self.split:
            self.subset = 'testing'
        else:
            raise ValueError('Irregular name of split! Should include \"train\", \"val\", or \"test\" to indicate its subset.')

        self.split_file = './data/splits/{}.txt'.format(split)

        if not os.path.exists(self.split_file):
            raise FileNotFoundError('Not found split file! Please include {}.txt in ./data/splits.'.format(self.split))

        self.pred_dir = pred_dir
        if self.pred_dir is not None:
            if self.split not in os.path.basename(os.path.abspath(self.pred_dir)):
                raise ValueError('Irregular name of prediction folder! Should include split name \"{}\" for consistency.'.format(self.split))

        with open(self.split_file, 'r') as f:
            self.sample_ids = [int(id) for id in f.read().splitlines()]

        self.subset_dir = os.path.join(self.root_dir , self.subset)
        self.image_dir = os.path.join(self.subset_dir, "image_2")
        self.label_dir = os.path.join(self.subset_dir, "label_2")
        self.calib_dir = os.path.join(self.subset_dir, "calib")

        self.depthpc_dir = os.path.join(self.subset_dir, "depth_pc")
        self.lidar_dir = os.path.join(self.subset_dir, "velodyne")
        self.depth_dir = os.path.join(self.subset_dir, "depth")

    def __len__(self):
        return len(self.sample_ids)

    def get_image(self, idx):
        idx = self.sample_ids[idx]
        img_filename = os.path.join(self.image_dir, "%06d.png" % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        idx = self.sample_ids[idx]
        lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % (idx))
        return utils.load_velo_scan(lidar_filename, dtype, n_vec)

    def get_calibration(self, idx):
        idx = self.sample_ids[idx]
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        idx = self.sample_ids[idx]
        if self.subset == "training":
            label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
            return utils.read_label(label_filename)
        else:
            print('WARNING: Testing set does not have label!')
            return None

    def get_pred_objects(self, idx):
        if self.pred_dir is None:
            raise RuntimeError('Prediction folder not provided!')
        idx = self.sample_ids[idx]
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        if os.path.exists(pred_filename):
            return utils.read_label(pred_filename)
        else:
            print('WARNING: Prediction file not found!')
            return None

    def get_depth(self, idx):
        idx = self.sample_ids[idx]
        img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
        return utils.load_depth(img_filename)


def show_image_with_boxes(img, calib, objects=[], objects_pred=[]):
    """ Show image with 2D bounding boxes """
    img_2d = np.copy(img)  # for 2d bbox
    img_3d = np.copy(img)  # for 3d bbox

    for obj in objects:
        if obj.type not in care_types:
            continue
        cv2.rectangle(
            img_2d,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 255, 0),
            2,
        )
        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
        img_3d = utils.draw_projected_box3d(img_3d, box3d_pts_2d)

    for obj in objects_pred:
        if obj.type not in care_types:
            continue
        cv2.rectangle(
            img_2d,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 0, 255),
            2,
        )
        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
        img_3d = utils.draw_projected_box3d(img_3d, box3d_pts_2d, color=(0, 0, 255))
    
    return img_2d, img_3d


def get_lidar_index_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    return fov_inds


def show_lidar_with_depth(
    pc_velo,
    calib,
    fig,
    objects=None,
    img_fov=False,
    img_width=None,
    img_height=None,
    objects_pred=None,
    depth=None,
    cam_img=None,
    constraint_box=False,
    pc_label=False,
    save=False,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    #print(("All point num: ", pc_velo.shape[0]))
    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height
        )
        pc_velo = pc_velo[pc_velo_index, :]
        #print(("FOV point num: ", pc_velo.shape))
    #print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig, pc_label=pc_label)

    # Draw depth
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))
        print("depth_pc_velo:", depth_pc_velo.shape)
        print("depth_pc_velo:", type(depth_pc_velo))
        print(depth_pc_velo[:5])
        draw_lidar(depth_pc_velo, fig=fig, pts_color=(1, 1, 1))

        if save:
            data_idx = 0
            vely_dir = "data/object/training/depth_pc"
            save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))
            print(save_filename)
            # np.save(save_filename+".npy", np.array(depth_pc_velo))
            depth_pc_velo = depth_pc_velo.astype(np.float32)
            depth_pc_velo.tofile(save_filename)

    color = (0, 1, 0)
    if objects is not None:
        for obj in objects:
            if obj.type not in care_types:
                continue
            # Draw gt 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)

            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color, label=obj.type)
            # Draw heading arrow
            _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )

    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type not in care_types:
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            #print("box3d_pts_3d_velo:")
            #print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    
    return fig