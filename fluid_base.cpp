#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>

template <typename T>
class Array2D {
public:
    int rows, cols;
    std::vector<T> data;
    Array2D() : rows(0), cols(0) {}
    Array2D(int r, int c, T init = T()) : rows(r), cols(c), data(r * c, init) {}
    inline T& operator()(int i, int j) {
        return data[i * cols + j];
    }
    inline const T& operator()(int i, int j) const {
        return data[i * cols + j];
    }
};

// Global constants and parameters
const double re = 70.0;        // Reynolds number
const double cfl = 0.2;         // CFL number
const int nlast = 10000;        // Number of time steps
const int nlp = 10;             // Interval for logging
const double omegap = 0.60;     // SOR parameter for pressure
const int maxitp = 199;         // Max iterations for pressure solver
const double errorp = 1.0e-4;   // Error tolerance for pressure solver
const double omegav = 1.20;     // SOR parameter for velocity
const int maxitv = 20;          // Max iterations for velocity solver
const double errorv = 1.0e-5;   // Error tolerance for velocity solver
double dt;

struct Grid {
    int mx, my;
    int i1, i2, j1, j2;
    double dx, dy;
    Array2D<double> x, y;
};

struct Flow {
    int nbegin;
    double time;
    Array2D<double> u, v, p;
    Flow(int mx, int my)
        : nbegin(0), time(0.0),
          u(mx + 2, my + 2, 0.0),
          v(mx + 2, my + 2, 0.0),
          p(mx + 2, my + 2, 0.0) {}
};

struct Worksp {
    Array2D<double> rhs, urhs, vrhs;
    Worksp(int mx, int my)
        : rhs(mx + 1, my + 1, 0.0),
          urhs(mx + 1, my + 1, 0.0),
          vrhs(mx + 1, my + 1, 0.0) {}
};

void setGrid(Grid &grid);
void solveFlow(Flow &flow, Grid &grid, Worksp &worksp, std::ofstream &hist_data, std::ofstream &flow_data);
void initializeConditions(Flow &flow, Grid &grid);
void solvePoissonEquation(Flow &flow, Grid &grid, Worksp &worksp, double &resp, int &itrp);
void solveVelocityEquation(Flow &flow, Grid &grid, Worksp &worksp);
void updateBoundaryConditionsForP(Flow &flow, Grid &grid);
void updateBoundaryConditionsForV(Flow &flow, Grid &grid);

int main() {
    std::ofstream flow_data("flow.data", std::ios::binary);
    if (!flow_data) {
        std::cerr << "Error opening flow.data" << std::endl;
        return 1;
    }
    std::ofstream hist_data("hist.data");
    if (!hist_data) {
        std::cerr << "Error opening hist.data" << std::endl;
        return 1;
    }

    Grid grid;
    setGrid(grid);

    // Construct flow field and working array according to grid size
    Flow flow(grid.mx, grid.my);
    Worksp worksp(grid.mx, grid.my);

    solveFlow(flow, grid, worksp, hist_data, flow_data);

    flow_data.close();
    hist_data.close();
    return 0;
}

void setGrid(Grid &grid) {
    grid.mx = 401;
    grid.i1 = 96;
    grid.i2 = 106;
    grid.dx = 1.0 / static_cast<double>(grid.i2 - grid.i1);

    grid.my = 201;
    grid.j1 = 96;
    grid.j2 = 106;
    grid.dy = 1.0 / static_cast<double>(grid.j2 - grid.j1);

    grid.x = Array2D<double>(grid.mx + 1, grid.my + 1, 0.0);
    grid.y = Array2D<double>(grid.mx + 1, grid.my + 1, 0.0);

    int icent = (grid.i1 + grid.i2) / 2;
    int jcent = (grid.j1 + grid.j2) / 2;
    for (int i = 1; i <= grid.mx; ++i) {
        for (int j = 1; j <= grid.my; ++j) {
            grid.x(i, j) = grid.dx * (i - icent);
            grid.y(i, j) = grid.dy * (j - jcent);
        }
    }
}

void solveFlow(Flow &flow, Grid &grid, Worksp &worksp, std::ofstream &hist_data, std::ofstream &flow_data) {
    double resp;
    int itrp;
    dt = cfl * std::min(grid.dx, grid.dy);
    std::cout << "*** Computation Conditions ***" << std::endl;
    std::cout << " CFL = " << cfl << std::endl;
    std::cout << " dt = " << dt << std::endl;
    std::cout << " " << nlast << " Time Steps to go..." << std::endl << std::endl;

    hist_data << ">> 2D Incompressible Flow Solver" << std::endl;
    hist_data << " Re " << re << std::endl;
    hist_data << " No. of Grid Points: " << grid.mx << " " << grid.my << std::endl;
    hist_data << " CFL / dt / Steps: " << cfl << " " << dt << " " << nlast << std::endl;

    std::cout << "u(96,95) = " << flow.u(96,95) << std::endl;
    dt = cfl * std::min(grid.dx, grid.dy);

    initializeConditions(flow, grid);
    updateBoundaryConditionsForP(flow, grid);
    updateBoundaryConditionsForV(flow, grid);

    std::cout << " Step / Res(p) / CD / CL / Cp1 / Cp2" << std::endl;
    hist_data << " Step Res(p) CD CL Cp1 Cp2" << std::endl;

    for (int n = 1; n <= nlast; ++n) {
        int nstep = n + flow.nbegin;
        flow.time += dt;

        // 1. Solving the Poisson equation for pressure
        solvePoissonEquation(flow, grid, worksp, resp, itrp);
        // 2. Update Pressure Boundary Condition
        updateBoundaryConditionsForP(flow, grid);
        // 3. Update velocity field
        solveVelocityEquation(flow, grid, worksp);
        // 4. Update speed boundary conditions
        updateBoundaryConditionsForV(flow, grid);
        // 5. Calculating lift and drag coefficients
        double cd = 0.0, cl = 0.0;
        double cp1, cp2;
        for (int j = grid.j1; j <= grid.j2 - 1; ++j) {
            double cpfore = (2.0 * flow.p(grid.i1, j) + 2.0 * flow.p(grid.i1, j + 1)) / 2.0;
            double cpback = (2.0 * flow.p(grid.i2, j) + 2.0 * flow.p(grid.i2, j + 1)) / 2.0;
            cd += (cpfore - cpback) * grid.dy;
        }
        for (int i = grid.i1; i <= grid.i2 - 1; ++i) {
            double cpbtm = (2.0 * flow.p(i, grid.j1) + 2.0 * flow.p(i + 1, grid.j1)) / 2.0;
            double cptop = (2.0 * flow.p(i, grid.j2) + 2.0 * flow.p(i + 1, grid.j2)) / 2.0;
            cl += (cpbtm - cptop) * grid.dx;
        }

        if (n % nlp == 0) {
            cp1 = 2.0 * flow.p(grid.i2 + grid.i2 - grid.i1, grid.j1);
            cp2 = 2.0 * flow.p(grid.i2 + grid.i2 - grid.i1, grid.j2);
            std::cout << nstep << " " << resp << " " << itrp << " " << cd << " " << cl << " " << cp1 << " " << cp2 << std::endl;
            hist_data << nstep << " " << resp << " " << cd << " " << cl << " " << cp1 << " " << cp2 << std::endl;
        }
    }

    for (int i = 1; i <= grid.mx; ++i) {
        for (int j = 1; j <= grid.my; ++j) {
            flow.p(i, j) *= 2.0;
        }
    }

    // Write flow field data
    flow_data << re << " " << cfl << " " << dt << " " << nlast << " " << flow.time << std::endl;
    flow_data << grid.mx << " " << grid.i1 << " " << grid.i2 << " " << grid.my << " " << grid.j1 << " " << grid.j2 << std::endl;

    // Write grid coordinates
    for (int j = 1; j <= grid.my; ++j) {
        for (int i = 1; i <= grid.mx; ++i) {
            flow_data << grid.x(i, j) << " " << grid.y(i, j) << " ";
        }
        flow_data << "\n";
    }

    // Write speed and pressure field data
    for (int j = 1; j <= grid.my; ++j) {
        for (int i = 1; i <= grid.mx; ++i) {
            flow_data << flow.u(i, j) << " " << flow.v(i, j) << " " << flow.p(i, j) << " ";
        }
        flow_data << "\n";
    }
}

void initializeConditions(Flow &flow, Grid &grid) {
    flow.nbegin = 0;
    flow.time = 0.0;
    for (int i = 1; i <= grid.mx; ++i) {
        for (int j = 1; j <= grid.my; ++j) {
            flow.u(i, j) = 1.0;
            flow.v(i, j) = 0.0;
            flow.p(i, j) = 0.0;
        }
    }
}

void solvePoissonEquation(Flow &flow, Grid &grid, Worksp &worksp, double &resp, int &itrp) {
    int mx = grid.mx;
    int my = grid.my;
    double dx = grid.dx;
    double dy = grid.dy;
    // Calculate the RHS
    for (int i = 2; i < mx; ++i) {
        for (int j = 2; j < my; ++j) {
            if (i > grid.i1 && i < grid.i2 && j > grid.j1 && j < grid.j2)
                continue;
            double ux = (flow.u(i + 1, j) - flow.u(i - 1, j)) / (2.0 * dx);
            double uy = (flow.u(i, j + 1) - flow.u(i, j - 1)) / (2.0 * dy);
            double vx = (flow.v(i + 1, j) - flow.v(i - 1, j)) / (2.0 * dx);
            double vy = (flow.v(i, j + 1) - flow.v(i, j - 1)) / (2.0 * dy);
            worksp.rhs(i, j) = (ux + vy) / dt - (ux * ux + 2.0 * uy * vx + vy * vy);
        }
    }

    double res;
    int itr;
    for (itr = 1; itr <= maxitp; ++itr) {
        res = 0.0;
        for (int i = 2; i <= mx - 1; ++i) {
            for (int j = 2; j <= my - 1; ++j) {
                if (i > grid.i1 && i < grid.i2 && j > grid.j1 && j < grid.j2)
                    continue;
                double term1 = (flow.p(i + 1, j) + flow.p(i - 1, j)) / (dx * dx);
                double term2 = (flow.p(i, j + 1) + flow.p(i, j - 1)) / (dy * dy);
                double term3 = worksp.rhs(i, j);
                double dp = term1 + term2 - term3;
                dp = dp / (2.0 / (dx * dx) + 2.0 / (dy * dy)) - flow.p(i, j);
                res += dp * dp;
                flow.p(i, j) += omegap * dp;
            }
        }
        updateBoundaryConditionsForP(flow, grid);
        res = std::sqrt(res / (mx * my));
        if (res < errorp)
            break;
    }
    resp = res;
    itrp = itr;
}

void solveVelocityEquation(Flow &flow, Grid &grid, Worksp &worksp) {
    int mx = grid.mx;
    int my = grid.my;
    double dx = grid.dx;
    double dy = grid.dy;
    
    // Pressure gradient term
    for (int i = 2; i < mx; ++i) {
        for (int j = 2; j < my; ++j) {
            if (i > grid.i1 && i < grid.i2 && j > grid.j1 && j < grid.j2)
                continue;
            worksp.urhs(i, j) = -(flow.p(i + 1, j) - flow.p(i - 1, j)) / (2.0 * dx);
            worksp.vrhs(i, j) = -(flow.p(i, j + 1) - flow.p(i, j - 1)) / (2.0 * dy);
        }
    }
    
    // Convection and Viscosity Terms (Part I: Second-Order Central Difference Calculation of Viscosity Terms)
    for (int i = 2; i < mx; ++i) {
        for (int j = 2; j < my; ++j) {
            if (i > grid.i1 && i < grid.i2 && j > grid.j1 && j < grid.j2)
                continue;
            worksp.urhs(i, j) += (flow.u(i + 1, j) - 2.0 * flow.u(i, j) + flow.u(i - 1, j)) / (re * dx * dx);
            worksp.urhs(i, j) += (flow.u(i, j + 1) - 2.0 * flow.u(i, j) + flow.u(i, j - 1)) / (re * dy * dy);
            worksp.vrhs(i, j) += (flow.v(i + 1, j) - 2.0 * flow.v(i, j) + flow.v(i - 1, j)) / (re * dx * dx);
            worksp.vrhs(i, j) += (flow.v(i, j + 1) - 2.0 * flow.v(i, j) + flow.v(i, j - 1)) / (re * dy * dy);
        }
    }
    
    // Convection discretization (update some boundary points using extrapolation method)
    for (int j = grid.j1 + 1; j <= grid.j2 - 1; ++j) {
        flow.u(grid.i1 + 1, j) = 2.0 * flow.u(grid.i1, j) - flow.u(grid.i1 - 1, j);
        flow.u(grid.i2 - 1, j) = 2.0 * flow.u(grid.i2, j) - flow.u(grid.i2 + 1, j);
        flow.v(grid.i1 + 1, j) = 2.0 * flow.v(grid.i1, j) - flow.v(grid.i1 - 1, j);
        flow.v(grid.i2 - 1, j) = 2.0 * flow.v(grid.i2, j) - flow.v(grid.i2 + 1, j);
    }
    
    // Convection and Viscosity Terms (Part II: Higher-Order Format Corrections)
    for (int i = 2; i <= mx - 1; ++i) {
        for (int j = 2; j <= my - 1; ++j) {
            if (i > grid.i1 && i < grid.i2 && j > grid.j1 && j < grid.j2)
                continue;
            worksp.urhs(i, j) = worksp.urhs(i, j)
                - flow.u(i, j) * (-flow.u(i + 2, j) + 8.0 * (flow.u(i + 1, j) - flow.u(i - 1, j)) + flow.u(i - 2, j)) / (12.0 * dx)
                - std::abs(flow.u(i, j)) * (flow.u(i + 2, j) - 4.0 * flow.u(i + 1, j) + 6.0 * flow.u(i, j) - 4.0 * flow.u(i - 1, j) + flow.u(i - 2, j)) / (4.0 * dx);
            worksp.vrhs(i, j) = worksp.vrhs(i, j)
                - flow.u(i, j) * (-flow.v(i + 2, j) + 8.0 * (flow.v(i + 1, j) - flow.v(i - 1, j)) + flow.v(i - 2, j)) / (12.0 * dx)
                - std::abs(flow.u(i, j)) * (flow.v(i + 2, j) - 4.0 * flow.v(i + 1, j) + 6.0 * flow.v(i, j) - 4.0 * flow.v(i - 1, j) + flow.v(i - 2, j)) / (4.0 * dx);
        }
    }

    for (int i = 2; i <= mx - 1; ++i) {
        for (int j = 2; j <= my - 1; ++j) {
            if (i > grid.i1 && i < grid.i2 && j > grid.j1 && j < grid.j2)
                continue;
            worksp.urhs(i, j) = worksp.urhs(i, j)
                - flow.v(i, j) * (-flow.u(i, j + 2) + 8.0 * (flow.u(i, j + 1) - flow.u(i, j - 1)) + flow.u(i, j - 2)) / (12.0 * dy)
                - std::abs(flow.v(i, j)) * (flow.u(i, j + 2) - 4.0 * flow.u(i, j + 1) + 6.0 * flow.u(i, j) - 4.0 * flow.u(i, j - 1) + flow.u(i, j - 2)) / (4.0 * dy);
            worksp.vrhs(i, j) = worksp.vrhs(i, j)
                - flow.v(i, j) * (-flow.v(i, j + 2) + 8.0 * (flow.v(i, j + 1) - flow.v(i, j - 1)) + flow.v(i, j - 2)) / (12.0 * dy)
                - std::abs(flow.v(i, j)) * (flow.v(i, j + 2) - 4.0 * flow.v(i, j + 1) + 6.0 * flow.v(i, j) - 4.0 * flow.v(i, j - 1) + flow.v(i, j - 2)) / (4.0 * dy);
        }
    }

    // Update velocity field
    for (int i = 2; i < grid.mx; ++i) {
        for (int j = 2; j < grid.my; ++j) {
            if (i > grid.i1 && i < grid.i2 && j > grid.j1 && j < grid.j2)
                continue;
            flow.u(i, j) += dt * worksp.urhs(i, j);
            flow.v(i, j) += dt * worksp.vrhs(i, j);
        }
    }
}

void updateBoundaryConditionsForP(Flow &flow, Grid &grid) {
    // Left boundry (i = 1)
    int i = 1;
    for (int j = 1; j <= grid.my; ++j) {
        flow.p(i, j) = 0.0;
    }
    // Right boundry (i = mx)
    i = grid.mx;
    for (int j = 1; j <= grid.my; ++j) {
        flow.p(i, j) = 0.0;
    }
    // Lower Boundary (j = 1)
    int j = 1;
    for (int i = 1; i <= grid.mx; ++i) {
        flow.p(i, j) = 0.0;
    }
    // Upper Boundary (j = my)
    j = grid.my;
    for (int i = 1; i <= grid.mx; ++i) {
        flow.p(i, j) = 0.0;
    }
    // Corner point condition
    flow.p(grid.i1, grid.j1) = flow.p(grid.i1 - 1, grid.j1 - 1);
    flow.p(grid.i1, grid.j2) = flow.p(grid.i1 - 1, grid.j2 + 1);
    flow.p(grid.i2, grid.j1) = flow.p(grid.i2 + 1, grid.j1 - 1);
    flow.p(grid.i2, grid.j2) = flow.p(grid.i2 + 1, grid.j2 + 1);
    // Left and right wall conditions
    i = grid.i1;
    for (j = grid.j1 + 1; j < grid.j2; ++j) {
        flow.p(i, j) = flow.p(i - 1, j);
    }
    i = grid.i2;
    for (j = grid.j1 + 1; j < grid.j2; ++j) {
        flow.p(i, j) = flow.p(i + 1, j);
    }
    // Upper and lower wall conditions
    j = grid.j1;
    for (i = grid.i1 + 1; i < grid.i2; ++i) {
        flow.p(i, j) = flow.p(i, j - 1);
    }
    j = grid.j2;
    for (i = grid.i1 + 1; i < grid.i2; ++i) {
        flow.p(i, j) = flow.p(i, j + 1);
    }
}

void updateBoundaryConditionsForV(Flow &flow, Grid &grid) {
    // Left boundary and adjacent ghost points (i = 1, 0)
    for (int j = 1; j <= grid.my; ++j) {
        flow.u(1, j) = 1.0;
        flow.v(1, j) = 0.0;
        flow.u(0, j) = 1.0;
        flow.v(0, j) = 0.0;
    }
    // Right boundary and external ghost points (i = mx, mx+1)
    for (int j = 1; j <= grid.my; ++j) {
        flow.u(grid.mx, j) = 2.0 * flow.u(grid.mx - 1, j) - flow.u(grid.mx - 2, j);
        flow.v(grid.mx, j) = 2.0 * flow.v(grid.mx - 1, j) - flow.v(grid.mx - 2, j);
        flow.u(grid.mx + 1, j) = 2.0 * flow.u(grid.mx, j) - flow.u(grid.mx - 1, j);
        flow.v(grid.mx + 1, j) = 2.0 * flow.v(grid.mx, j) - flow.v(grid.mx - 1, j);
    }
    // (j = 1)
    for (int i = 1; i <= grid.mx; ++i) {
        flow.u(i, 1) = 2.0 * flow.u(i, 2) - flow.u(i, 3);
        flow.v(i, 1) = 2.0 * flow.v(i, 2) - flow.v(i, 3);
        flow.u(i, 0) = 2.0 * flow.u(i, 1) - flow.u(i, 2);
        flow.v(i, 0) = 2.0 * flow.v(i, 1) - flow.v(i, 2);
    }
    // (j = my, my+1)
    for (int i = 1; i <= grid.mx; ++i) {
        flow.u(i, grid.my) = 2.0 * flow.u(i, grid.my - 1) - flow.u(i, grid.my - 2);
        flow.v(i, grid.my) = 2.0 * flow.v(i, grid.my - 1) - flow.v(i, grid.my - 2);
        flow.u(i, grid.my + 1) = 2.0 * flow.u(i, grid.my) - flow.u(i, grid.my - 1);
        flow.v(i, grid.my + 1) = 2.0 * flow.v(i, grid.my) - flow.v(i, grid.my - 1);
    }
    // Fixed wall
    for (int i = grid.i1; i <= grid.i2; ++i) {
        for (int j = grid.j1; j <= grid.j2; ++j) {
            flow.u(i, j) = 0.0;
            flow.v(i, j) = 0.0;
        }
    }
}
