import matplotlib.pyplot as plt
import numpy as np

# State Variables - [Current position in X, velocity in meters]
variables_count = 2
# Observations
X = np.array([4000, 4260, 4550, 4860, 5110])  # in meters
V = np.array([280, 282, 285, 286, 290])  # in meters/second
Acc = np.array([2, 2, 2, 2, 2])  # in meters/second^2
sample_size = len(X)
# Process Errors in Process Covariance Matrix
del_Px = 20  # in meters
del_Pv = 5  # in meters/second

# Observation Errors
del_x = 25  # in meters
del_v = 6  # in meters/second

# Variables for easier mapping
del_t = 1
del_t2 = 1

# State Matrix for velocity measurements - X = A*X(k-1) + B*Acc + wk
X_p = np.zeros(shape=[sample_size, 2, 1])  # predicted state
X_p[0] = np.array([[X[0]], [V[0]]])

# Estimated State Matrix
X_k = np.zeros(shape=[sample_size, 2, 1])  # estimated state
X_k[0] = np.array([[X[0]], [V[0]]]) # adding first value as the same as observation

# State Matrix calculation constants
A = np.array([[1, del_t], [0, 1]])
B = np.array([[0.5 * del_t2], [del_t]])
wk = 0  # predicted state noise ~ 0, for easier calculations

# Prediceted and Estimated Process Covariance Matrix and adding initial state
P_p = np.zeros(shape=[sample_size, 2, 2])
P_k = np.zeros(shape=[sample_size, 2, 2])

P_p[0] = np.array(([del_Px ** 2, 0], [0, del_Pv ** 2])).reshape([1, 2, 2])
Q = 0  # process noise will be 0, to make life easier.

# Kalman Gain Paramaters
H = np.array([[1, 0], [0, 1]])  # only to make sure kalman filter parameters are in correct shape
R = np.array(([del_x, 0], [0, del_v])).reshape([1, 2, 2])  # sensor noise matrix

# Current Observations
Y = np.array([([element_1], [element_2])for element_1, element_2 in zip(X, V)])
C = np.array([[1, 0], [0, 1]])  # observation constants
Z = 0  # error in the process used for observations

for index in range(1, len(X)):
    # State Matrix
    U_k = np.array([[Acc[index]], [0]])

    X_p[index] = np.dot(A, X_p[index - 1]) + B * U_k + wk
    # Fun Fact: *, np.dot, and np.multiply yield different results at different times, and are very confusing and
    # amusing to debug at 3:30 AM.

    # Calculation of Covariance matrix  P_k = (np.dot(A, P), A.transpose()) + Q
    P_p[index] = np.dot(np.dot(A, P_p[index - 1]), A.T) + Q

    # Calculation of Kalman Gain
    K = np.divide(np.dot(P_p[index], H.T), np.dot(np.dot(H, P_p[index]), H.T) + R)

    # Current Observation
    Y[index] = np.dot(C, Y[index]) + Z

    # Estimating the state
    X_k[index] = X_p[index] + np.dot(K, np.subtract(Y[index], np.dot(H, X_p[index])))

    # Updating the covariance matrix
    P_k = np.dot(np.subtract(np.identity(variables_count), np.dot(K, H)), P_p[index])

print("Done")


# Plot observed values and kalman values
plt.plot(X, '-o', c='red')
plt.plot(X_k[:][:, 0], '-o', c='green')
plt.show()