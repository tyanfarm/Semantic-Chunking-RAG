import numpy as np

S = np.array([
    [1, 0.9746, -0.3780],
    [0.9746, 1, -0.1612],
    [-0.3780, -0.1612, 1]
])

mean_value = np.mean(S)
print(mean_value)  # Output: 0.2222