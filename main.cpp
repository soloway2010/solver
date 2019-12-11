#include <iostream>
#include <cstring>
#include <cmath>
#include <omp.h>

using namespace std;

void generateMatrix(int Nx, int Ny, int Nz, int (*Col)[7], double (*Val)[7])
{
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                int index = k * (Nx * Ny) + j * Ny + i;
                int* colRow = Col[index];
                double* valRow = Val[index];

                if (k > 0) {
                    colRow[0] = index - Nx * Ny;
                    valRow[0] = cos(index * (index - Nx * Ny) + 3.14);
                } else {
                    colRow[0] = -1;
                    valRow[0] = 0;
                }

                if (j > 0) {
                    colRow[1] = index - Nx;
                    valRow[1] = cos(index * (index - Nx) + 3.14);
                } else {
                    colRow[1] = -1;
                    valRow[1] = 0;
                }

                if (i > 0) {
                    colRow[2] = index - 1;
                    valRow[2] = cos(index * (index - 1) + 3.14);
                } else {
                    colRow[2] = -1;
                    valRow[2] = 0;
                }

                if (i < Nx - 1) {
                    colRow[4] = index + 1;
                    valRow[4] = cos(index * (index + 1) + 3.14);
                } else {
                    colRow[4] = -1;
                    valRow[4] = 0;
                }

                if (j < Ny - 1) {
                    colRow[5] = index + Nx;
                    valRow[5] = cos(index * (index + Nx) + 3.14);
                } else {
                    colRow[5] = -1;
                    valRow[5] = 0;
                }

                if (k < Nz - 1) {
                    colRow[6] = index + Nx * Ny;
                    valRow[6] = cos(index * (index + Nx * Ny) + 3.14);
                } else {
                    colRow[6] = -1;
                    valRow[6] = 0;
                }

                colRow[3] = index;
                valRow[3] = 1.5 * (abs(valRow[0]) + abs(valRow[1]) + abs(valRow[2]) + abs(valRow[4]) + abs(valRow[5]) + abs(valRow[6]));
            }
        }
    }
}

double dotSeq(double* x, double* y, int N)
{
    double prod = 0;

    for (int i = 0; i < N; i++) {
        prod += x[i] * y[i];
    }

    return prod;
}

double dotPar(double* x, double* y, int N)
{
    double prod = 0;

    #pragma omp parallel for reduction (+:prod)
    for (int i = 0; i < N; i++) {
        prod += x[i] * y[i];
    }

    return prod;
}

void axpbySeq(double a, double* x, double b, double* y, double* z, int N)
{
    for (int i = 0; i < N; i++) {
        z[i] = a * x[i] + b * y[i];
    }
}

void axpbyPar(double a, double* x, double b, double* y, double* z, int N)
{
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        z[i] = a * x[i] + b * y[i];
    }
}

void SpMVSeq(double* x, int (*Col)[7], double (*Val)[7], double* z, int N)
{
    for (int i = 0; i < N; i++) {
        z[i] = 0;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 7; j++) {
            if (Col[i][j] != -1) {
                z[Col[i][j]] += Val[i][j] * x[i];
            }
        }
    }
}

void SpMVPar(double* x, int (*Col)[7], double (*Val)[7], double* z, int N)
{
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        z[i] = 0;
    }

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 7; j++) {
            if (Col[i][j] != -1) {
                z[Col[i][j]] += Val[i][j] * x[i];
            }
        }
    }
}

void test(bool par, int Nx, int Ny, int Nz)
{
    int N = Nx * Ny * Nz;
    double a = 0.3;
    double b = 0.7;
    double* x = new double[N];
    double* y = new double[N];
    double* z = new double[N];
    int (*Col)[7] = new int[N][7];
    double (*Val)[7] = new double[N][7];

    double dot;
    double dotTime;
    double axpbySum = 0;
    double axpbyL2 = 0;
    double axpbyTime;
    double SpMVSum = 0;
    double SpMVL2 = 0;
    double SpMVTime;

    // fill in test data
    generateMatrix(Nx, Ny, Nz, Col, Val);

    for (int i = 0; i < N; i++) {
        x[i] = cos(i * i);
        y[i] = sin(i * i);
    }

    cout << "***** " << (par ? "Par" : "Seq") << " test *****" << endl;
    cout << "==== Start ====" << endl;

    // ordinary operations testing
    if (!par) {
        // seq dot
        dotTime = omp_get_wtime();
        dot = dotSeq(x, y, N);
        dotTime = omp_get_wtime() - dotTime;

        // seq axpby
        axpbyTime = omp_get_wtime();
        axpbySeq(a, x, b, y, z, N);
        axpbyTime = omp_get_wtime() - axpbyTime;

        for (int i = 0; i < N; i++) {
            axpbySum += z[i];
            axpbyL2 += z[i] * z[i];
        }

        axpbyL2 = sqrt(axpbyL2);

        // seq SpMV
        SpMVTime = omp_get_wtime();
        SpMVSeq(x, Col, Val, z, N);
        SpMVTime = omp_get_wtime() - SpMVTime;
        for (int i = 0; i < N; i++) {
            SpMVSum += z[i];
            SpMVL2 += z[i] * z[i];
        }

        SpMVL2 = sqrt(SpMVL2);
    } else {
        // par dot
        dotTime = omp_get_wtime();
        dot = dotPar(x, y, N);
        dotTime = omp_get_wtime() - dotTime;

        // par axpby
        axpbyTime = omp_get_wtime();
        axpbyPar(a, x, b, y, z, N);
        axpbyTime = omp_get_wtime() - axpbyTime;
        for (int i = 0; i < N; i++) {
            axpbySum += z[i];
            axpbyL2 += z[i] * z[i];
        }

        axpbyL2 = sqrt(axpbyL2);

        // par SpMV
        SpMVTime = omp_get_wtime();
        SpMVPar(x, Col, Val, z, N);
        SpMVTime = omp_get_wtime() - SpMVTime;
        for (int i = 0; i < N; i++) {
            SpMVSum += z[i];
            SpMVL2 += z[i] * z[i];
        }

        SpMVL2 = sqrt(SpMVL2);
    }

    cout << "dot: " << dot << endl;
    cout << "dot - time: " << dotTime << endl;
    cout << endl;
    cout << "axpby - sum: " << axpbySum << endl;
    cout << "axpby - L2: " << axpbyL2 << endl;
    cout << "axpby - time: " << axpbyTime << endl;
    cout << endl;
    cout << "SpMV - sum: " << SpMVSum << endl;
    cout << "SpMV - L2: " << SpMVL2 << endl;
    cout << "SpMV - time: " << SpMVTime << endl;

    cout << "==== End ====" << endl;

    // clean up
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] Col;
    delete[] Val;
}

void solveSeq(int Nx, int Ny, int Nz, double tol, int maxit, bool qa)
{
    int N = Nx * Ny * Nz;

    int (*Col)[7] = new int[N][7];
    double (*Val)[7] = new double[N][7];
    double* x = new double[N];
    double* b = new double[N];

    double* p = new double[N];
    double* q = new double[N];
    double* z = new double[N];
    double* r = new double[N];
    int (*ColM)[7] = new int[N][7];
    double (*ValM)[7] = new double[N][7];

    double alpha;
    double beta;
    double ro;
    double roOld;

    int i;
    double time;

    if (qa) {
        test(false, Nx, Ny, Nz);
    }

    // gen left part
    generateMatrix(Nx, Ny, Nz, Col, Val);

    // gen right part, nullify first guess and fill in M
    for (int i = 0; i < N; i++) {
        x[i] = 0;
        b[i] = cos(i);

        for (int j = 0; j < 7; j++) {
            if (j == 3) {
                ColM[i][j] = i;
                ValM[i][j] = 1 / Val[i][j];
            } else {
                ColM[i][j] = -1;
                ValM[i][j] = 0;
            }
        }
    }

    // main algorithm
    SpMVSeq(x, Col, Val, r, N);
    axpbySeq(1, b, -1, r, r, N);

    time = omp_get_wtime();
    for (i = 1; i <= maxit; i++) {
        SpMVSeq(r, ColM, ValM, z, N);
        roOld = ro;
        ro = dotSeq(r, z, N);

        if (i == 1) {
            axpbySeq(0, z, 1, z, p, N);
        } else {
            beta = ro / roOld;
            axpbySeq(1, z, beta, p, p, N);
        }

        SpMVSeq(p, Col, Val, q, N);
        alpha = ro / dotSeq(p, q, N);
        axpbySeq(1, x, alpha, p, x, N);
        axpbySeq(1, r, -alpha, q, r, N);

        if (ro < tol) {
            break;
        }
    }
    time = omp_get_wtime() - time;

    cout << "***** Seq Result *****" << endl;
    cout << "Main cycle time: " << time << endl;
    cout << "Iterations: " << i << endl;
    cout << "Error: " << ro << endl;

    // clean up
    delete[] Col;
    delete[] Val;
    delete[] ColM;
    delete[] ValM;
    delete[] b;
    delete[] x;
    delete[] p;
    delete[] q;
    delete[] z;
    delete[] r;
}

void solvePar(int Nx, int Ny, int Nz, double tol, int maxit, bool qa)
{
    int N = Nx * Ny * Nz;

    int (*Col)[7] = new int[N][7];
    double (*Val)[7] = new double[N][7];
    double* x = new double[N];
    double* b = new double[N];

    double* p = new double[N];
    double* q = new double[N];
    double* z = new double[N];
    double* r = new double[N];
    int (*ColM)[7] = new int[N][7];
    double (*ValM)[7] = new double[N][7];

    double alpha;
    double beta;
    double ro;
    double roOld;

    int i;
    double time;

    if (qa) {
        test(true, Nx, Ny, Nz);
    }

    // gen left part
    generateMatrix(Nx, Ny, Nz, Col, Val);

    // gen right part, nullify first guess and fill in M
    for (int i = 0; i < N; i++) {
        x[i] = 0;
        b[i] = cos(i);

        for (int j = 0; j < 7; j++) {
            if (j == 3) {
                ColM[i][j] = i;
                ValM[i][j] = 1 / Val[i][j];
            } else {
                ColM[i][j] = -1;
                ValM[i][j] = 0;
            }
        }
    }

    // main algorithm
    SpMVPar(x, Col, Val, r, N);
    axpbyPar(1, b, -1, r, r, N);

    time = omp_get_wtime();
    for (i = 1; i <= maxit; i++) {
        SpMVPar(r, ColM, ValM, z, N);
        roOld = ro;
        ro = dotPar(r, z, N);

        if (i == 1) {
            axpbyPar(0, z, 1, z, p, N);
        } else {
            beta = ro / roOld;
            axpbyPar(1, z, beta, p, p, N);
        }

        SpMVPar(p, Col, Val, q, N);
        alpha = ro / dotPar(p, q, N);
        axpbyPar(1, x, alpha, p, x, N);
        axpbyPar(1, r, -alpha, q, r, N);

        if (ro < tol) {
            break;
        }
    }
    time = omp_get_wtime() - time;

    cout << "***** Par Result *****" << endl;
    cout << "Main cycle time: " << time << endl;
    cout << "Iterations: " << i << endl;
    cout << "Error: " << ro << endl;

    // clean up
    delete[] Col;
    delete[] Val;
    delete[] ColM;
    delete[] ValM;
    delete[] b;
    delete[] x;
    delete[] p;
    delete[] q;
    delete[] z;
    delete[] r;
} 

int main(int argc, char** argv)
{
    int Nx;
    int Ny;
    int Nz;
    double tol;
    int maxit;
    int nt;
    bool qa = false;

    if (argc < 13) {
        cout << "-nx <int> -ny <int> -nz <int>" << endl;
        cout << "-tol <double>" << endl;
        cout << "-maxit <int>" << endl;
        cout << "-nt <int>" << endl;
        cout << "-qa" << endl;

        return 0;
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-nx")) {
            Nx = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-ny")) {
            Ny = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-nz")) {
            Nz = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-tol")) {
            tol = atof(argv[++i]);
        } else if (!strcmp(argv[i], "-maxit")) {
            maxit = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-nt")) {
            nt = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-qa")) {
            qa = true;
        }
    }

    omp_set_num_threads(nt);
    solveSeq(Nx, Ny, Nz, tol, maxit, qa);
    cout << endl;
    solvePar(Nx, Ny, Nz, tol, maxit, qa);

    return 0;
}