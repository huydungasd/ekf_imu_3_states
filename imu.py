import numpy as np
import pandas as pd
import math

from time import sleep
from scipy.spatial.transform import Rotation as R


class IMU:
        
    def __init__(self, imu_path, gt_path):
        self.imu_data = pd.read_csv(imu_path).to_numpy(dtype=float)
        self.gt_data = pd.read_csv(gt_path).to_numpy(dtype=float)
    
    def get_t(self):
        return self.imu_data[:, 0]         
        
    # rad/s
    def get_gyro(self):
        return [self.imu_data[:, 7], self.imu_data[:, 8], self.imu_data[:, 9]]        

    def get_mag(self):
        return [self.imu_data[:, 4], self.imu_data[:, 5], self.imu_data[:, 6]]    
        
    # m/s^2
    def get_acc(self):
        return [self.imu_data[:, 1], self.imu_data[:, 2], self.imu_data[:, 3]]

    def get_postion(self):
        return [self.gt_data[:, 1], self.gt_data[:, 2], self.gt_data[:, 3]]

    def get_orientation(self, angle=False):
        if not angle:
            return [self.gt_data[:, 5], self.gt_data[:, 6], self.gt_data[:, 7], self.gt_data[:, 4]]
        else:
            angles = R.from_quat(np.hstack([self.gt_data[:, 5].reshape((-1, 1)), self.gt_data[:, 6].reshape((-1, 1)).reshape((-1, 1)), self.gt_data[:, 7].reshape((-1, 1)), self.gt_data[:, 4].reshape((-1, 1))])).as_euler(seq='xyz')
            return [angles[:, 0], angles[:, 1], angles[:, 2]]
    
    # rad
    def get_acc_angles(self):
        [ax, ay, az] = self.get_acc()
        phi = np.arctan2(ay, np.sqrt(ax ** 2.0 + az ** 2.0))
        theta = np.arctan2(-ax, np.sqrt(ay ** 2.0 + az ** 2.0))
        return [phi, theta]

def get_mag_yaw(mag_x, mag_y, mag_z, prev_angle):
    a = -mag_y * np.cos(prev_angle[0]) + mag_z * np.sin(prev_angle[0])
    b = mag_x * np.cos(prev_angle[1])
    c = mag_y * np.sin(prev_angle[1]) * np.sin(prev_angle[0])
    d = mag_z * np.sin(prev_angle[1]) * np.cos(prev_angle[0])
    
    return (np.arctan2(a,  (b + c + d)) + np.pi/2)