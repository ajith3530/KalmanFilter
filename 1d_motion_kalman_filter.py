import numpy as np

# Observations
X = np.array([4000, 4260, 4550, 4860, 5110])  # in meters
V = np.array([280, 282, 285, 286, 290])  # in meters/second
Acc = np.array([2, 2, 2, 2, 2])  # in meters/second^2

# Process Errors in Process Covariance Matrix
del_Px = 20  # in meters
del_Pv = 5  # in meters/second

# Observation Errors
del_x = 25  # in meters
del_v = 6  # in meters/second

# Variables for easier mapping
del_t = 1
del_t2 = 1

# State Matrix for velocity measurements - X = A*X_prev + B*Acc + wk
A = np.array([[1, del_t], [0, 1]])
B = np.array([[0.5 * del_t2], [del_t]])
wk = 0  # predicted state noise ~ 0, for easier calculatins
# Process Covariance Matrix
P_init = np.array()
for index in range(1, len(X)):
    # State Matrix
    U_k = np.array([[Acc[index]], [0]])
    X_prev = np.array([[X[index - 1]], [V[index - 1]]])
    X_new = np.dot(A, X_prev) + B * U_k + wk
    # Fun Fact: *, np.dot, and np.multiply yield different results, and are very confusing.

    # Calculation of Covariance matrix  P_k = (np.dot(A, P), A.transpose()) + Q
