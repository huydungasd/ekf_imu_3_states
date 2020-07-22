import numpy as np

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def eulerAnglesToRotationMatrix(theta) :
    """
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = R_z @ R_y @ R_x
    """
    return R.from_euler('xyz', theta).as_matrix()


# Lissage des signaux
def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.zeros(values.shape)
    sma[:window-1] = values[:window-1]
    sma[window-1:] = np.convolve(values, weights, 'valid')
    return sma
    

def quaternion_to_matrix(q):
    return R.from_quat(q).as_matrix()


def quaternion_to_euler(q):
    return R.from_quat(q).as_euler(seq='xyz')


def update_orientation_quaternion(prev_q, angular_rate, dt):
    q1 = R.from_quat(prev_q)
    q2 = R.from_euler(seq='xyz', angles=angular_rate*dt)
    q_updated = q1 * q2

    return q_updated.as_quat()


def nominal_state_predict(prev_state, input_data, dt, g):
    # Receive information
    p, v, q = prev_state[:3], prev_state[3:6], prev_state[6:10]
    a_m, omega_m = input_data[0:3], input_data[3:6]

    # Prediction
    p_predict = p + v * dt + 0.5 * (quaternion_to_matrix(q) @ a_m + g) * dt**2
    v_predict = v + (quaternion_to_matrix(q) @ a_m + g) * dt
    q_predict = update_orientation_quaternion(q, omega_m, dt)

    return np.concatenate([p_predict, v_predict, q_predict])


def skew(omega):
    assert omega.shape == (3,)
    return np.array([[  0,          -omega[2],  omega[1]    ],
                     [  omega[2],   0,          -omega[0]   ],
                     [  -omega[1],  omega[0],   0           ]])

def error_state_predict(prev_delta_x, prev_P_delta_x, x, input_data, dt, V_i, Theta_i):
    # Receive information
    p, v, q = x[:3], x[3:6], x[6:10]
    a_m, omega_m = input_data[0:3], input_data[3:6]

    # Fx, Fi, Qi matrices - equation 270, page 61, https://arxiv.org/pdf/1711.02508.pdf
    Fx = np.identity(9)
    R_matrix = quaternion_to_matrix(q)
    Fx[:3, 3:6] = np.eye(3) * dt
    Fx[3:6, 6:9] = -skew(R_matrix @ a_m) * dt

    Fi = np.zeros((9, 6))
    Fi[3:6,0:3] = R_matrix * dt
    Fi[6:9,3:6] = -R_matrix * dt

    Qi = np.zeros((6, 6))
    Qi[:3, :3] = V_i
    Qi[3:6, 3:6] = Theta_i

    delta_x_predict = Fx @ prev_delta_x
    P_delta_x_predict = Fx @ prev_P_delta_x @ Fx.T + Fi @ Qi @ Fi.T

    return delta_x_predict, P_delta_x_predict


def zero_velocity_update(x, P_delta_x, V, velo):
    # Receive information
    p, v, q, a_b, omega_b, g = x[:3], x[3:6], x[6:10], x[10:13], x[13:16], x[16:19]

    # Section 6.1, https://arxiv.org/pdf/1711.02508.pdf

    Q_delta_theta = 0.5 * np.array([[   -q[0],  -q[1],  -q[2]   ],
                                    [   q[3],   q[2],  -q[1]    ],
                                    [   -q[2],   q[3],   q[0]   ],
                                    [   q[1],  -q[0],   q[3]    ]])
    
    X_delta_x = np.zeros((10, 9))
    X_delta_x[:6, :6] = np.eye(6)
    X_delta_x[6:10, 6:9] = Q_delta_theta

    H_x = np.zeros((3, 10))
    H_x[:3, 3:6] = np.eye(3)


    H = H_x @ X_delta_x
    H = np.zeros((3,9))
    H[:, 3:6] = np.identity(3)

    K = P_delta_x @ H.T @ np.linalg.inv((H @ P_delta_x @ H.T + V))
    delta_x_update = K @ (np.array([*velo]) - H_x @ x)

    P_delta_x_update = (np.eye(9) - K @ H) @ P_delta_x @ (np.eye(9) - K @ H).T + K @ V @ K.T

    return delta_x_update, P_delta_x_update


def injection_obs_err_to_nominal_state(x, delta_x):
    # Receive information
    p, v, q = x[:3], x[3:6], x[6:10]
    delta_p, delta_v, delta_theta = delta_x[:3], delta_x[3:6], delta_x[6:9]

    # Section 6.2, https://arxiv.org/pdf/1711.02508.pdf
    p_update = p + delta_p
    v_update = v + delta_v
    # q_update = update_orientation_quaternion(q, delta_theta, 1)
    R_matrix = quaternion_to_matrix(q)
    omega = np.array([[0,-delta_x[8], delta_x[7]],[delta_x[8],0,-delta_x[6]],[-delta_x[7],delta_x[6],0]])
    R_matrix = (np.identity(3) + omega) @ R_matrix
    q_update = R.from_matrix(R_matrix).as_quat()

    return np.concatenate([p_update, v_update, q_update])



def ESKF_reset(delta_x, P_delta_x):
    # Receive information
    delta_theta = delta_x[6:9]

    # Section 6.3, https://arxiv.org/pdf/1711.02508.pdf
    delta_x_update = np.zeros((9,))
    
    G = np.identity(9)
    G[6:9, 6:9] = np.eye(3) + skew(0.5 * delta_theta)
    
    P_delta_x_update = G @ P_delta_x @ G.T

    return delta_x_update, P_delta_x_update


def SHOE(imudata, g, W=5, G=4.1e8, sigma_a=0.00098**2, sigma_w=(8.7266463e-5)**2):
    T = np.zeros(np.int(np.floor(imudata.shape[0]/W)+1))
    zupt = np.zeros(imudata.shape[0])
    a = np.zeros((1,3))
    w = np.zeros((1,3))
    inv_a = 1/sigma_a
    inv_w = 1/sigma_w
    acc = imudata[:,0:3]
    gyro = imudata[:,3:6]

    i=0
    for k in range(0,imudata.shape[0]-W+1,W): #filter through all imu readings
        smean_a = np.mean(acc[k:k+W,:],axis=0)
        for s in range(k,k+W):
            a.put([0,1,2],acc[s,:])
            w.put([0,1,2],gyro[s,:])
            T[i] += inv_a*( (a - g * smean_a/np.linalg.norm(smean_a)).dot(( a - g * smean_a/np.linalg.norm(smean_a)).T)) #acc terms
            T[i] += inv_w*( (w).dot(w.T) )
        zupt[k:k+W].fill(T[i])
        i+=1
    zupt = zupt/W
    plt.figure()
    plt.plot(zupt)
    return zupt < G