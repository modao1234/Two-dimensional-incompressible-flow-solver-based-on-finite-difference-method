import numpy as np
import matplotlib.pyplot as plt

def main():
    filename = 'flow.data'
    with open(filename, 'r') as f:

        line = f.readline().strip()
        global_params = list(map(float, line.split()))
        re, cfl, dt, nlast, sim_time = global_params

        line = f.readline().strip()
        grid_params = list(map(int, line.split()))
        mx, i1, i2, my, j1, j2 = grid_params

        X = np.zeros((my, mx))
        Y = np.zeros((my, mx))
        for j in range(my):
            line = f.readline().strip()
            nums = list(map(float, line.split()))
            for i in range(mx):
                X[j, i] = nums[2 * i]
                Y[j, i] = nums[2 * i + 1]

        U = np.zeros((my, mx))
        V = np.zeros((my, mx))
        P = np.zeros((my, mx))
        for j in range(my):
            line = f.readline().strip()
            nums = list(map(float, line.split()))
            for i in range(mx):
                U[j, i] = nums[3 * i]
                V[j, i] = nums[3 * i + 1]
                P[j, i] = nums[3 * i + 2]

    speed = np.sqrt(U**2 + V**2)

    i1_py = i1 - 1
    i2_py = i2 - 1
    j1_py = j1 - 1
    j2_py = j2 - 1

    x1, y1 = X[j1_py, i1_py], Y[j1_py, i1_py]
    x2, y2 = X[j1_py, i2_py], Y[j1_py, i2_py]
    x3, y3 = X[j2_py, i2_py], Y[j2_py, i2_py]
    x4, y4 = X[j2_py, i1_py], Y[j2_py, i1_py]

    boundary_x = [x1, x2, x3, x4, x1]
    boundary_y = [y1, y2, y3, y4, y1]

    plt.figure(figsize=(10, 8))
    strm = plt.streamplot(X, Y, U, V, color=speed, cmap='viridis', density=2, linewidth=1)
    plt.colorbar(strm.lines, label='Speed')
    plt.title('Streamlines with Speed Color')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(boundary_x, boundary_y, 'k-', linewidth=2)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("streamline.png", dpi=300)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, P, levels=50, cmap='jet')
    plt.colorbar(contour, label='Pressure')
    plt.title('Pressure Contour with Boundary')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(boundary_x, boundary_y, 'k-', linewidth=2)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("pressure_contour.png", dpi=300)

    plt.show()

if __name__ == '__main__':
    main()
