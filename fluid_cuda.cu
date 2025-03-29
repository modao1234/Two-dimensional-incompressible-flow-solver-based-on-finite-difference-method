#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

#define IDX2D(i, j, ncols) ((i)*(ncols) + (j))

const double re = 70.0;      
const double cfl = 0.2;       
const int nlast = 10000;     
const int nlp = 10;           
const double omegap = 0.60;    
const int maxitp = 199;       
const double errorp = 1.0e-4;  

const int MX = 401;
const int MY = 201;
const int I1 = 96;
const int I2 = 106;
const int J1 = 96;
const int J2 = 106;
const double dx = 1.0 / double(I2 - I1);
const double dy = 1.0 / double(J2 - J1);

const int NX = MX + 2;
const int NY = MY + 2;

// Kernel：Initialization conditions
__global__ void init_conditions(double *u, double *v, double *p, int mx, int my) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // i = 1..mx
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // j = 1..my
    if(i <= mx && j <= my) {
        int idx = IDX2D(i, j, my+2);
        u[idx] = 1.0;
        v[idx] = 0.0;
        p[idx] = 0.0;
    }
}

// Kernel：Update Boundary Conditions For P
__global__ void bc_P_inflow_downstream(double *p, int my) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // j=1..my
    if(j <= my) {
        // Inflow: i = 1
        p[IDX2D(1, j, my+2)] = 0.0;
        // Downstream: i = MX
        p[IDX2D(MX, j, my+2)] = 0.0;
    }
}

__global__ void bc_P_bottom_top(double *p, int mx, int my) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // i=1..mx
    if(i <= mx) {
        // Bottom: j = 1
        p[IDX2D(i, 1, my+2)] = 0.0;
        // Top: j = MY
        p[IDX2D(i, my, my+2)] = 0.0;
    }
}

//    p[I1][J1] = p[I1-1][J1-1], p[I1][J2] = p[I1-1][J2+1],
//    p[I2][J1] = p[I2+1][J1-1], p[I2][J2] = p[I2+1][J2+1]
__global__ void bc_P_corners(double *p, int my) {
    if(threadIdx.x == 0 && blockIdx.x == 0) {
        p[IDX2D(I1, J1, my+2)] = p[IDX2D(I1-1, J1-1, my+2)];
        p[IDX2D(I1, J2, my+2)] = p[IDX2D(I1-1, J2+1, my+2)];
        p[IDX2D(I2, J1, my+2)] = p[IDX2D(I2+1, J1-1, my+2)];
        p[IDX2D(I2, J2, my+2)] = p[IDX2D(I2+1, J2+1, my+2)];
    }
}

//    for j = J1+1 .. J2-1: p[I1][j] = p[I1-1][j], p[I2][j] = p[I2+1][j]
__global__ void bc_P_left_right(double *p, int my) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + J1 + 1;
    if(j < J2) {
        p[IDX2D(I1, j, my+2)] = p[IDX2D(I1-1, j, my+2)];
        p[IDX2D(I2, j, my+2)] = p[IDX2D(I2+1, j, my+2)];
    }
}

//    for i = I1+1 .. I2-1: p[i][J1] = p[i][J1-1], p[i][J2] = p[i][J2+1]
__global__ void bc_P_top_bottom(double *p, int my) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + I1 + 1;
    if(i < I2) {
        p[IDX2D(i, J1, my+2)] = p[IDX2D(i, J1-1, my+2)];
        p[IDX2D(i, J2, my+2)] = p[IDX2D(i, J2+1, my+2)];
    }
}

// Kernel：Calculate the right hand side of the pressure Poisson 
__global__ void compute_rhs(const double *u, const double *v, double *rhs,
                              int mx, int my, int i1, int i2, int j1, int j2,
                              double dx, double dy, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 2;
    if(i <= mx - 1 && j <= my - 1) {
        if ((i > i1 && i < i2) && (j > j1 && j < j2)) return;
        int idx = IDX2D(i, j, my+2);
        double ux = (u[IDX2D(i+1, j, my+2)] - u[IDX2D(i-1, j, my+2)]) / (2.0 * dx);
        double uy = (u[IDX2D(i, j+1, my+2)] - u[IDX2D(i, j-1, my+2)]) / (2.0 * dy);
        double vx = (v[IDX2D(i+1, j, my+2)] - v[IDX2D(i-1, j, my+2)]) / (2.0 * dx);
        double vy = (v[IDX2D(i, j+1, my+2)] - v[IDX2D(i, j-1, my+2)]) / (2.0 * dy);
        rhs[idx] = (ux + vy) / dt - (ux * ux + 2.0 * uy * vx + vy * vy);
    }
}

// Kernel：Single-step Poisson iteration
__global__ void poisson_iteration(double *p, const double *rhs,
                                  int mx, int my, int i1, int i2, int j1, int j2,
                                  double dx, double dy, double omegap, double *d_sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 2;
    if(i <= mx - 1 && j <= my - 1) {
        if ((i > i1 && i < i2) && (j > j1 && j < j2)) return;
        int idx = IDX2D(i, j, my+2);
        double term1 = (p[IDX2D(i+1, j, my+2)] + p[IDX2D(i-1, j, my+2)]) / (dx * dx);
        double term2 = (p[IDX2D(i, j+1, my+2)] + p[IDX2D(i, j-1, my+2)]) / (dy * dy);
        double term3 = rhs[idx];
        double dp = term1 + term2 - term3;
        dp = dp / (2.0/(dx*dx) + 2.0/(dy*dy)) - p[idx];
        p[idx] += omegap * dp;
        atomicAdd(d_sum, dp * dp);
    }
}

// Kernel：Calculate the right side of the velocity equation
__global__ void compute_velocity_rhs(const double *p, double *urhs, double *vrhs,
                                     int mx, int my, int i1, int i2, int j1, int j2,
                                     double dx, double dy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 2;
    if(i <= mx - 1 && j <= my - 1) {
        if ((i > i1 && i < i2) && (j > j1 && j < j2)) return;
        int idx = IDX2D(i, j, my+2);
        urhs[idx] = -(p[IDX2D(i+1, j, my+2)] - p[IDX2D(i-1, j, my+2)]) / (2.0 * dx);
        vrhs[idx] = -(p[IDX2D(i, j+1, my+2)] - p[IDX2D(i, j-1, my+2)]) / (2.0 * dy);
    }
}

// Kernel：Higher-order convection correction - x-direction part
__global__ void convection_correction_x(const double *u, const double *v, double *urhs, double *vrhs,
                                          int mx, int my, double dx, int i1, int i2, int j1, int j2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 2;
    if(i <= mx - 1 && j <= my - 1) {
        if ((i > i1 && i < i2) && (j > j1 && j < j2)) return;
        int idx = IDX2D(i, j, my+2);
        double u_ij = u[idx];

        double term1 = (-u[IDX2D(i+2, j, my+2)] + 8.0*(u[IDX2D(i+1, j, my+2)] - u[IDX2D(i-1, j, my+2)]) + u[IDX2D(i-2, j, my+2)])/(12.0*dx);
        double term2 = (u[IDX2D(i+2, j, my+2)] - 4.0*u[IDX2D(i+1, j, my+2)] + 6.0*u_ij - 4.0*u[IDX2D(i-1, j, my+2)] + u[IDX2D(i-2, j, my+2)])/(4.0*dx);
        double corr_u = u_ij * term1 + fabs(u_ij) * term2;

        double term3 = (-v[IDX2D(i+2, j, my+2)] + 8.0*(v[IDX2D(i+1, j, my+2)] - v[IDX2D(i-1, j, my+2)]) + v[IDX2D(i-2, j, my+2)])/(12.0*dx);
        double term4 = (v[IDX2D(i+2, j, my+2)] - 4.0*v[IDX2D(i+1, j, my+2)] + 6.0*v[idx] - 4.0*v[IDX2D(i-1, j, my+2)] + v[IDX2D(i-2, j, my+2)])/(4.0*dx);
        double corr_v = u_ij * term3 + fabs(u_ij) * term4;

        urhs[idx] -= corr_u;
        vrhs[idx] -= corr_v;
    }
}

// Kernel：Higher-order convection correction - y-direction part
__global__ void convection_correction_y(const double *u, const double *v, double *urhs, double *vrhs,
                                          int mx, int my, double dy, int i1, int i2, int j1, int j2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 2;
    if(i <= mx - 1 && j <= my - 1) {
        if ((i > i1 && i < i2) && (j > j1 && j < j2)) return;
        int idx = IDX2D(i, j, my+2);

        double term1 = (-u[IDX2D(i, j+2, my+2)] + 8.0*(u[IDX2D(i, j+1, my+2)] - u[IDX2D(i, j-1, my+2)]) + u[IDX2D(i, j-2, my+2)])/(12.0*dy);
        double term2 = (u[IDX2D(i, j+2, my+2)] - 4.0*u[IDX2D(i, j+1, my+2)] + 6.0*u[idx] - 4.0*u[IDX2D(i, j-1, my+2)] + u[IDX2D(i, j-2, my+2)])/(4.0*dy);
        double corr_u = v[idx] * term1 + fabs(v[idx]) * term2;

        double term3 = (-v[IDX2D(i, j+2, my+2)] + 8.0*(v[IDX2D(i, j+1, my+2)] - v[IDX2D(i, j-1, my+2)]) + v[IDX2D(i, j-2, my+2)])/(12.0*dy);
        double term4 = (v[IDX2D(i, j+2, my+2)] - 4.0*v[IDX2D(i, j+1, my+2)] + 6.0*v[idx] - 4.0*v[IDX2D(i, j-1, my+2)] + v[IDX2D(i, j-2, my+2)])/(4.0*dy);
        double corr_v = v[idx] * term3 + fabs(v[idx]) * term4;
        urhs[idx] -= corr_u;
        vrhs[idx] -= corr_v;
    }
}

// Kernel：Update the velocity field, calculate the viscosity term (central difference) and add the right-hand side term  
__global__ void update_velocity(double *u, double *v, const double *urhs, const double *vrhs,
                                int mx, int my, double dx, double dy, double dt,
                                int i1, int i2, int j1, int j2, double re)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 2;
    if(i <= mx - 1 && j <= my - 1) {
        if ((i > i1 && i < i2) && (j > j1 && j < j2)) return;
        int idx = IDX2D(i, j, my+2);
        double visc_u = (u[IDX2D(i+1, j, my+2)] - 2.0*u[idx] + u[IDX2D(i-1, j, my+2)])/(re*dx*dx)
                        + (u[IDX2D(i, j+1, my+2)] - 2.0*u[idx] + u[IDX2D(i, j-1, my+2)])/(re*dy*dy);
        double visc_v = (v[IDX2D(i+1, j, my+2)] - 2.0*v[idx] + v[IDX2D(i-1, j, my+2)])/(re*dx*dx)
                        + (v[IDX2D(i, j+1, my+2)] - 2.0*v[idx] + v[IDX2D(i, j-1, my+2)])/(re*dy*dy);
        u[idx] += dt * (urhs[idx] + visc_u);
        v[idx] += dt * (vrhs[idx] + visc_v);
    }
}

// Kernel：Update Boundary Conditions For V
__global__ void bc_V_inflow(double *u, double *v, int my) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if(j <= my) {
        u[IDX2D(1,j, my+2)] = 1.0;
        v[IDX2D(1,j, my+2)] = 0.0;
        u[IDX2D(0,j, my+2)] = 1.0;
        v[IDX2D(0,j, my+2)] = 0.0;
    }
}

__global__ void bc_V_downstream(double *u, double *v, int my) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if(j <= my) {
        u[IDX2D(MX, j, my+2)] = 2.0*u[IDX2D(MX-1, j, my+2)] - u[IDX2D(MX-2, j, my+2)];
        v[IDX2D(MX, j, my+2)] = 2.0*v[IDX2D(MX-1, j, my+2)] - v[IDX2D(MX-2, j, my+2)];
        u[IDX2D(MX+1, j, my+2)] = 2.0*u[IDX2D(MX, j, my+2)] - u[IDX2D(MX-1, j, my+2)];
        v[IDX2D(MX+1, j, my+2)] = 2.0*v[IDX2D(MX, j, my+2)] - v[IDX2D(MX-1, j, my+2)];
    }
}

__global__ void bc_V_bottom(double *u, double *v, int mx, int my) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if(i <= mx) {
        u[IDX2D(i, 1, my+2)] = 2.0*u[IDX2D(i, 2, my+2)] - u[IDX2D(i, 3, my+2)];
        v[IDX2D(i, 1, my+2)] = 2.0*v[IDX2D(i, 2, my+2)] - v[IDX2D(i, 3, my+2)];
        u[IDX2D(i, 0, my+2)] = 2.0*u[IDX2D(i, 1, my+2)] - u[IDX2D(i, 2, my+2)];
        v[IDX2D(i, 0, my+2)] = 2.0*v[IDX2D(i, 1, my+2)] - v[IDX2D(i, 2, my+2)];
    }
}

__global__ void bc_V_top(double *u, double *v, int mx, int my) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if(i <= mx) {
        u[IDX2D(i, my, my+2)] = 2.0*u[IDX2D(i, my-1, my+2)] - u[IDX2D(i, my-2, my+2)];
        v[IDX2D(i, my, my+2)] = 2.0*v[IDX2D(i, my-1, my+2)] - v[IDX2D(i, my-2, my+2)];
        u[IDX2D(i, my+1, my+2)] = 2.0*u[IDX2D(i, my, my+2)] - u[IDX2D(i, my-1, my+2)];
        v[IDX2D(i, my+1, my+2)] = 2.0*v[IDX2D(i, my, my+2)] - v[IDX2D(i, my-1, my+2)];
    }
}

__global__ void bc_V_obstacle(double *u, double *v, int my) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + I1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + J1;
    if(i <= I2 && j <= J2) {
        int idx = IDX2D(i, j, my+2);
        u[idx] = 0.0;
        v[idx] = 0.0;
    }
}

int main() {
    double dt = cfl * std::min(dx, dy);

    size_t size = NX * NY * sizeof(double);
    double *d_u, *d_v, *d_p;
    double *d_rhs, *d_urhs, *d_vrhs;
    CUDA_CHECK(cudaMalloc((void**)&d_u, size));
    CUDA_CHECK(cudaMalloc((void**)&d_v, size));
    CUDA_CHECK(cudaMalloc((void**)&d_p, size));
    CUDA_CHECK(cudaMalloc((void**)&d_rhs, size));
    CUDA_CHECK(cudaMalloc((void**)&d_urhs, size));
    CUDA_CHECK(cudaMalloc((void**)&d_vrhs, size));

    double *d_sum;
    CUDA_CHECK(cudaMalloc((void**)&d_sum, sizeof(double)));

    dim3 blockDim(16,16);
    dim3 gridDim((MX+blockDim.x-1)/blockDim.x, (MY+blockDim.y-1)/blockDim.y);

    // 1. Initialize u, v, p (interior region)
    init_conditions<<<gridDim, blockDim>>>(d_u, d_v, d_p, MX, MY);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. Update pressure boundary conditions: call each kernel separately
    int threads1D = 256;
    int grid_1D_j = (MY + threads1D - 1)/threads1D;
    bc_P_inflow_downstream<<<grid_1D_j, threads1D>>>(d_p, MY);
    int grid_1D_i = (MX + threads1D - 1)/threads1D;
    bc_P_bottom_top<<<grid_1D_i, threads1D>>>(d_p, MX, MY);
    bc_P_corners<<<1,1>>>(d_p, MY);
    int grid_1D_wall = ((J2 - J1 - 1) + threads1D - 1)/threads1D;
    bc_P_left_right<<<grid_1D_wall, threads1D>>>(d_p, MY);
    int grid_1D_topbot = ((I2 - I1 - 1) + threads1D - 1)/threads1D;
    bc_P_top_bottom<<<grid_1D_topbot, threads1D>>>(d_p, MY);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. At the same time, the velocity boundary conditions are updated
    int grid_1D_v = (MY + threads1D - 1)/threads1D;
    bc_V_inflow<<<grid_1D_v, threads1D>>>(d_u, d_v, MY);
    bc_V_downstream<<<grid_1D_v, threads1D>>>(d_u, d_v, MY);
    int grid_1D_bottom = (MX + threads1D - 1)/threads1D;
    bc_V_bottom<<<grid_1D_bottom, threads1D>>>(d_u, d_v, MX, MY);
    bc_V_top<<<grid_1D_bottom, threads1D>>>(d_u, d_v, MX, MY);
    dim3 blockDim2(16,16);
    dim3 gridDim2(((I2-I1+1)+blockDim2.x-1)/blockDim2.x, ((J2-J1+1)+blockDim2.y-1)/blockDim2.y);
    bc_V_obstacle<<<gridDim2, blockDim2>>>(d_u, d_v, MY);
    CUDA_CHECK(cudaDeviceSynchronize());

    double time = 0.0;
    int nstep;
    for (int n = 1; n <= nlast; n++) {
        nstep = n;
        time += dt;

        // 4. Calculate the right side of the Poisson equation
        compute_rhs<<<gridDim, blockDim>>>(d_u, d_v, d_rhs,
            MX, MY, I1, I2, J1, J2, dx, dy, dt);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 5. Poisson iteration
        int itrp;
        double h_sum = 0.0, residual = 1e10;
        for (itrp = 1; itrp <= maxitp; itrp++) {
            CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
            poisson_iteration<<<gridDim, blockDim>>>(d_p, d_rhs,
                MX, MY, I1, I2, J1, J2, dx, dy, omegap, d_sum);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
            residual = sqrt(h_sum / (MX * MY));
            if(residual < errorp) break;
        }
        // Update the pressure boundary condition again
        bc_P_inflow_downstream<<<grid_1D_j, threads1D>>>(d_p, MY);
        bc_P_bottom_top<<<grid_1D_i, threads1D>>>(d_p, MX, MY);
        bc_P_corners<<<1,1>>>(d_p, MY);
        bc_P_left_right<<<grid_1D_wall, threads1D>>>(d_p, MY);
        bc_P_top_bottom<<<grid_1D_topbot, threads1D>>>(d_p, MY);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 6. Calculate the right side of the velocity equation (pressure gradient part)
        compute_velocity_rhs<<<gridDim, blockDim>>>(d_p, d_urhs, d_vrhs,
            MX, MY, I1, I2, J1, J2, dx, dy);
        CUDA_CHECK(cudaDeviceSynchronize());
        // 7. Apply higher-order convection corrections: first in the x direction, then in the y direction
        convection_correction_x<<<gridDim, blockDim>>>(d_u, d_v, d_urhs, d_vrhs,
            MX, MY, dx, I1, I2, J1, J2);
        CUDA_CHECK(cudaDeviceSynchronize());
        convection_correction_y<<<gridDim, blockDim>>>(d_u, d_v, d_urhs, d_vrhs,
            MX, MY, dy, I1, I2, J1, J2);
        CUDA_CHECK(cudaDeviceSynchronize());
        // 8. Update velocity field: add viscosity term
        update_velocity<<<gridDim, blockDim>>>(d_u, d_v, d_urhs, d_vrhs,
            MX, MY, dx, dy, dt, I1, I2, J1, J2, re);
        CUDA_CHECK(cudaDeviceSynchronize());
        // Update speed boundary conditions
        bc_V_inflow<<<grid_1D_v, threads1D>>>(d_u, d_v, MY);
        bc_V_downstream<<<grid_1D_v, threads1D>>>(d_u, d_v, MY);
        bc_V_bottom<<<grid_1D_bottom, threads1D>>>(d_u, d_v, MX, MY);
        bc_V_top<<<grid_1D_bottom, threads1D>>>(d_u, d_v, MX, MY);
        bc_V_obstacle<<<gridDim2, blockDim2>>>(d_u, d_v, MY);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 9. Calculating lift and drag coefficients
        double cd = 0.0, cl = 0.0, cp1 = 0.0, cp2 = 0.0;
        double *h_p = new double[NX * NY];
        CUDA_CHECK(cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost));

        for (int j = J1; j <= J2 - 1; j++) {
            double cpfore = (2.0 * h_p[IDX2D(I1, j, NY)] + 2.0 * h_p[IDX2D(I1, j+1, NY)]) / 2.0;
            double cpback = (2.0 * h_p[IDX2D(I2, j, NY)] + 2.0 * h_p[IDX2D(I2, j+1, NY)]) / 2.0;
            cd += (cpfore - cpback) * dy;
        }

        for (int i = I1; i <= I2 - 1; i++) {
            double cpbtm = (2.0 * h_p[IDX2D(i, J1, NY)] + 2.0 * h_p[IDX2D(i+1, J1, NY)]) / 2.0;
            double cptop = (2.0 * h_p[IDX2D(i, J2, NY)] + 2.0 * h_p[IDX2D(i+1, J2, NY)]) / 2.0;
            cl += (cpbtm - cptop) * dx;
        }
        cp1 = 2.0 * h_p[IDX2D(I2 + (I2 - I1), J1, NY)];
        cp2 = 2.0 * h_p[IDX2D(I2 + (I2 - I1), J2, NY)];
        delete[] h_p;

        if(n % nlp == 0) {
            std::cout << nstep << " " << residual << " " << itrp << " " << cd << " " << cl << " " << cp1 << " " << cp2 << std::endl;
        }
    }

    double *h_p_final = new double[NX * NY];
    CUDA_CHECK(cudaMemcpy(h_p_final, d_p, size, cudaMemcpyDeviceToHost));
    for (int i = 1; i <= MX; i++) {
        for (int j = 1; j <= MY; j++) {
            h_p_final[IDX2D(i,j, NY)] *= 2.0;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_p, h_p_final, size, cudaMemcpyHostToDevice));
    delete[] h_p_final;

    double *h_u_final = new double[NX * NY];
    double *h_v_final = new double[NX * NY];
    double *h_p_final2 = new double[NX * NY];
    CUDA_CHECK(cudaMemcpy(h_u_final, d_u, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v_final, d_v, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_p_final2, d_p, size, cudaMemcpyDeviceToHost));
    std::ofstream flow_data("flow.data", std::ios::binary);
    if(!flow_data) {
        std::cerr << "Error opening output file" << std::endl;
        exit(1);
    }

    flow_data << re << " " << cfl << " " << dt << " " << nlast << " " << time << "\n";

    flow_data << MX << " " << I1 << " " << I2 << " " << MY << " " << J1 << " " << J2 << "\n";

    for (int j = 1; j <= MY; j++) {
        for (int i = 1; i <= MX; i++) {
            double x = dx * (i - (I1 + I2)/2.0);
            double y = dy * (j - (J1 + J2)/2.0);
            flow_data << x << " " << y << " ";
        }
        flow_data << "\n";
    }

    for (int j = 1; j <= MY; j++) {
        for (int i = 1; i <= MX; i++) {
            flow_data << h_u_final[IDX2D(i,j, NY)] << " "
                      << h_v_final[IDX2D(i,j, NY)] << " "
                      << h_p_final2[IDX2D(i,j, NY)] << " ";
        }
        flow_data << "\n";
    }
    flow_data.close();

    delete[] h_u_final;
    delete[] h_v_final;
    delete[] h_p_final2;

    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_urhs));
    CUDA_CHECK(cudaFree(d_vrhs));
    CUDA_CHECK(cudaFree(d_sum));

    return 0;
}
