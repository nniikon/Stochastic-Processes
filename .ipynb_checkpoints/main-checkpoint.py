import numpy as np
import matplotlib.pyplot as plt

def wiener_process(T = 1.0, N = 1000, seed=None):
    np.random.seed(seed)

    dt = T / N
    t = np.linspace(0, T, N+1)
    dW = np.sqrt(dt) * np.random.randn(N)

    W = np.zeros(N+1)
    W[1:] = np.cumsum(dW)

    return t, W

t, W = wiener_process(T=1.0, N=1000, seed=42)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(t, W)
plt.title('Wiener Process Simulation')
plt.xlabel('Time')
plt.ylabel('W(t)')
plt.grid(True)
plt.show()
