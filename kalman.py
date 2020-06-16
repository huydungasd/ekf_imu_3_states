import os
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from time import sleep, time
from math import sin, cos, tan, pi

from utils import *
from imu import *

data_folder = 4
fname = 3
dir_path = os.path.dirname(os.path.realpath(__file__))
gt_path = os.path.join(dir_path, f'data{data_folder}/gt', f'{fname}.csv')
imu_path = os.path.join(dir_path, f'data{data_folder}/imu', f'{fname}.csv')
imu = IMU(imu_path, gt_path)

"""
++++++++++++++++++++++++++      Get raw data      ++++++++++++++++++++++++++        
"""
# Get acceleration measurements
[acc_x, acc_y, acc_z] = imu.get_acc()
# acc_x = movingaverage(acc_x, 10)
# acc_y = movingaverage(acc_y, 10)
# acc_z = movingaverage(acc_z, 10)

# Get gyro measurements
[gyr_x, gyr_y, gyr_z] = imu.get_gyro()
# gyr_x = movingaverage(gyr_x, 10)
# gyr_y = movingaverage(gyr_y, 10)
# gyr_z = movingaverage(gyr_z, 10)

# Get mag measurements
[mag_x, mag_y, mag_z] = imu.get_mag()
# mag_x = movingaverage(mag_x, 10)
# mag_y = movingaverage(mag_y, 10)
# mag_z = movingaverage(mag_z, 10)


"""
++++++++++++++++++++++++++      Get ground truth      ++++++++++++++++++++++++++
"""
# Get position
[x_gt, y_gt, z_gt] = imu.get_postion()
# Get orientation
[qx_gt, qy_gt, qz_gt, qw_gt] = imu.get_orientation()
[phi, theta, gamma] = imu.get_orientation(angle=True)



"""
++++++++++++++++++++++++++      Estimate orientation from acceleration (confident only when the sensor is stable)      ++++++++++++++++++++++++++
"""
[phi_acc, theta_acc] = imu.get_acc_angles()
# phi_acc = movingaverage(phi_acc, window=10)
# theta_acc = movingaverage(theta_acc, window=10)

# Calculate accelerometer offsets
N = 50
phi_offset, theta_offset, gamma_offset, phi_true_offset, theta_true_offset, gamma_true_offset = 0, 0, 0, 0, 0, 0
for i in range(N):
    phi_offset += phi_acc[i]
    theta_offset += theta_acc[i]
    gamma_offset += get_mag_yaw(mag_x[i], mag_y[i], mag_z[i], np.array([phi[i], theta[i]]))
    phi_true_offset += phi[i]
    theta_true_offset += theta[i]
    gamma_true_offset += gamma[i]

phi_offset = phi_offset / N - phi_true_offset / N
theta_offset = theta_offset / N - theta_true_offset / N
gamma_offset = gamma_offset / N - gamma_true_offset / N

print("Roll, Pitch calculated by acceleration offset: " + str(phi_offset) + "," + str(theta_offset))
print("Yaw calculated by magnetometer offset: " + str(gamma_offset))
sleep(1)

# Get accelerometer measurements and remove offsets
phi_acc -= phi_offset
theta_acc -= theta_offset


"""
++++++++++++++++++++++++++      Initialisation      ++++++++++++++++++++++++++
"""
print("Running...")
t = imu.get_t()
t = t - t[0]

p_0 = np.array([x_gt[0], y_gt[0], z_gt[0]])
v_0 = np.zeros((3,))
q_0 = R.from_quat(np.array([qx_gt[0], qy_gt[0], qz_gt[0], qw_gt[0]])).as_quat()
g = np.array([0, 0, -9.8029])

V_i_0 = np.identity(3) * 0.5**2
Theta_i_0 = np.identity(3) * (0.5 * np.pi / 180)**2
V = np.identity(3) * 1e-4

x_hats = np.zeros((imu.imu_data.shape[0] - 1, 10))
delta_x_hats  = np.zeros((imu.imu_data.shape[0] - 1, 9))

x = np.concatenate([p_0, v_0, q_0])
delta_x = np.zeros((9,))
P_theta_x = np.identity(9)
P_theta_x[0:3,0:3] = 1e-10*np.identity(3)
P_theta_x[3:6,3:6] = 1e-10*np.identity(3)
P_theta_x[6:9,6:9] = (0.1*np.pi/180)**2*np.identity(3)

shoe_detector = SHOE(imu.imu_data[:, [1,2,3,7,8,9]], g)

list1 = []
list2 = []
for i in range(imu.imu_data.shape[0] - 1):
    # Sampling time
    dt = t[i+1] - t[i]
    input_data = np.array([acc_x[i+1], acc_y[i+1], acc_z[i+1], gyr_x[i+1], gyr_y[i+1], gyr_z[i+1]])
    
    # Prediction
    x = nominal_state_predict(x, input_data, dt, g)
    delta_x, P_theta_x = error_state_predict(delta_x, P_theta_x, x, input_data, dt, V_i_0, Theta_i_0)

    if shoe_detector[i+1]:
        list1.append(i)
        # angle_xy = np.array([phi_acc[i], theta_acc[i]])
        # angle_z = get_mag_yaw(mag_x[i], mag_y[i], mag_z[i], angle_xy) - gamma_offset
        # quat_from_acc = R.from_euler(seq='xyz', angles=np.array([angle_xy[0], angle_xy[1], angle_z])).as_quat()
        delta_x, P_theta_x = zero_velocity_update(x, P_theta_x, V, np.array([0, 0, 0]))
        x = injection_obs_err_to_nominal_state(x, delta_x)
        delta_x, P_theta_x = ESKF_reset(delta_x, P_theta_x)
    else:
        list2.append(i)
        delta_x, P_theta_x = zero_velocity_update(x, P_theta_x, V, np.array([-1.21049284742314,-0.088205541585698,-0.528302390620874])*0.5)
        x = injection_obs_err_to_nominal_state(x, delta_x)
        delta_x, P_theta_x = ESKF_reset(delta_x, P_theta_x)
    x_hats[i, :] = x

    delta_x_hats[i, :] = delta_x

angles = quaternion_to_euler(x_hats[:, 6:10])
phi_hat, theta_hat, gamma_hat = angles[:, 0], angles[:, 1], angles[:, 2]
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Display results
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(t[:-1], phi_hat, 'b', label='$\hat{\phi}$ (Prediction)')
axs[0].plot(t, phi, 'c', label='$\phi$ by internal algo of IMU sensor')
# axs[0].plot(t, phi_interg, 'g', label='$\phi_{interg}$ (Intergration)')
axs[0].set_ylabel('Rad')
axs[0].legend()

axs[1].plot(t[list1], theta_hat[list1], '.b', label='$\hat{\\theta}$ (Prediction)')
axs[1].plot(t[list2], theta_hat[list2], '.r', label='$\hat{\\theta}$ (Prediction)')
axs[1].plot(t, theta, 'c', label='$\\theta$ by internal algo of IMU sensor')
# axs[1].plot(t, theta_interg, 'g', label='$\\theta_{interg}$ (Intergration)')
axs[1].set_ylabel('Rad')
axs[1].legend()

axs[2].plot(t[:-1], gamma_hat, 'b', label='$\hat{\gamma}$ (Prediction)')
axs[2].plot(t, gamma, 'c', label='$\gamma$ by internal algo of IMU sensor')
# axs[2].plot(t, gamma_interg, 'g', label='$\gamma_{interg}$ (Intergration)')
axs[2].set_ylabel('Rad')
axs[2].set_xlabel('Time')
axs[2].legend()

plt.figure()
plt.gca(projection='3d')
plt.plot(x_hats[:, 0], x_hats[:, 1], x_hats[:, 2], '.b')
plt.plot(x_hats[list2, 0], x_hats[list2, 1], x_hats[list2, 2], '.r')
plt.plot(x_gt, y_gt, z_gt)
plt.show()