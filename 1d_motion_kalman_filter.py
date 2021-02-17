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
Xp = np.zeros(shape=[sample_size, 2, 1])  # predicted state
Xp[0] = np.array([[X[0]], [V[0]]])

# Estimated State Matrix
Xk = np.zeros(shape=[sample_size, 2, 1])  # estimated state
Xk[0] = np.array([[X[0]], [V[0]]]) # adding first value as the same as observation

# State Matrix calculation constants
A = np.array([[1, del_t], [0, 1]])
B = np.array([[0.5 * del_t2], [del_t]])
wk = 0  # predicted state noise ~ 0, for easier calculations

# Prediceted and Estimated Process Covariance Matrix and adding initial state
Pp = np.zeros(shape=[sample_size, 2, 2])
Pk = np.zeros(shape=[sample_size, 2, 2])

Pp[0] = np.array(([del_Px ** 2, 0], [0, del_Pv ** 2])).reshape([1, 2, 2])
Q = 0  # process noise will be 0, to make life easier.

# Kalman Gain Paramaters
H = np.array([[1, 0], [0, 1]])  # only to make sure kalman filter parameters are in correct shape
R = np.array(([del_x**2, 0], [0, del_v**2]))  # sensor noise matrix

# Current Observations
Y = np.array([([element_1], [element_2])for element_1, element_2 in zip(X, V)])
C = np.array([[1, 0], [0, 1]])  # observation constants
Z = 0  # error in the process used for observations

for index in range(1, len(X)):
    # State Matrix
    U_k = np.array([[Acc[index]]])

    Xp[index] = np.dot(A, Xp[index - 1]) + np.dot(B, U_k) + wk
    # Fun Fact: *, np.dot, and np.multiply yield different results at different times, and are very confusing and
    # amusing to debug at 3:30 AM.

    # Calculation of Covariance matrix  P_k = (np.dot(A, P), A.transpose()) + Q
    Pp[index] = np.dot(np.dot(A, Pp[index - 1]), A.T) + Q
    Pp[index][0, 1] = 0
    Pp[index][1, 0] = 0

    # Calculation of Kalman Gain
    K = np.divide(np.dot(Pp[index], H.T), np.dot(np.dot(H, Pp[index]), H.T) + R)
    K[np.isnan(K)] = 0

    # Current Observation
    Y[index] = np.dot(C, Y[index]) + Z

    # Estimating the state
    Xk[index] = Xp[index] + np.dot(K, np.subtract(Y[index], np.dot(H, Xp[index])))

    # Updating the covariance matrix
    Pk = np.dot(np.subtract(np.identity(variables_count), np.dot(K, H)), Pp[index])

# Plot observed values and kalman values
plt.plot(X, '-o', c='red')
plt.plot(Xk[:][:, 0], '-o', c='green')
plt.show()