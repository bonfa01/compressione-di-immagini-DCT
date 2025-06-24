import numpy as np

def dct_base(f):
    D = np.zeros((f, f))
    for i in range(f):
        for j in range(f):
            if i == 0:
                D[i, j] = 1 / np.sqrt(f)
            else:
                D[i, j] = np.sqrt(2 / f) * np.cos((np.pi * (2 * j + 1) * i) / (2 * f))
    return D

def dct(x, D):
    return D @ x

def idct(c, D):
    return D.T @ c

def dct2(block, D):
    return D @ block @ D.T

def idct2(block, D):
    return D.T @ block @ D
