import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from codice.dct_utils import dct_base, dct2, idct2 


def dct2_scipy(matrix):
    return dctn(matrix, norm='ortho')

def idct2_scipy(matrix):
    return idctn(matrix, norm='ortho')

def benchmark_dct(sizes, ripetizioni=5):
    tempi_custom = []
    tempi_scipy = []

    for N in sizes:
        print(f"Benchmarking N={N}")
        A = np.random.rand(N, N).astype(np.float32)

        # Tempo medio per scipy
        t_scipy = 0
        for _ in range(ripetizioni):
            B = A.copy()
            start = time.perf_counter()
            B = dct2_scipy(B)
            B[N//4:, N//4:] = 0
            idct2_scipy(B)
            t_scipy += time.perf_counter() - start
        tempi_scipy.append(t_scipy / ripetizioni)

        # Tempo medio per custom
        D = dct_base(N)
        t_custom = 0
        for _ in range(ripetizioni):
            C = A.copy()
            start = time.perf_counter()
            C = dct2(C, D)
            C[N//4:, N//4:] = 0
            idct2(C, D)
            t_custom += time.perf_counter() - start
        tempi_custom.append(t_custom / ripetizioni)

    return tempi_scipy, tempi_custom


def plot_benchmark(sizes, tempi_scipy, tempi_custom):
    n = np.array(sizes)
    nlogn = n**2 * np.log2(n)
    n3 = n**3

    # Normalizzazione delle curve teoriche
    scale_nlogn = tempi_scipy[0] / nlogn[0]
    scale_n3 = tempi_custom[0] / n3[0]
    curve_nlogn = scale_nlogn * nlogn
    curve_n3 = scale_n3 * n3

    plt.figure(figsize=(12, 7))
    plt.semilogy(sizes, tempi_scipy, 'bo-', label='scipy')
    plt.semilogy(sizes, tempi_custom, 'ro-', label='custom')

    plt.xlabel('Dimensione matrice')
    plt.ylabel('Tempo medio (s)')
    plt.title('grafico sulla media di 5 esecuzioni')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    os.makedirs('risultati', exist_ok=True)
    plt.savefig('risultati/confronto_esteso.jpeg', dpi=800)
    plt.tight_layout()
    plt.show()


def main():
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    tempi_scipy, tempi_custom = benchmark_dct(sizes, ripetizioni=5)
    plot_benchmark(sizes, tempi_scipy, tempi_custom)


if __name__ == "__main__":
    main()
