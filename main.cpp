#include "iostream"
#include "cmath"

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

void axpbySeq(double a, double* x, double b, double* y, double* z, int N)
{
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
    double axpbySum = 0;
    double axpbyL2 = 0;
    double SpMVSum = 0;
    double SpMVL2 = 0;

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
        dot = dotSeq(x, y, N);

        // seq axpby
        axpbySeq(a, x, b, y, z, N);
        for (int i = 0; i < N; i++) {
            axpbySum += z[i];
            axpbyL2 += z[i] * z[i];
        }

        axpbyL2 = sqrt(axpbyL2);

        // seq SpMV
        SpMVSeq(x, Col, Val, z, N);
        for (int i = 0; i < N; i++) {
            SpMVSum += z[i];
            SpMVL2 += z[i] * z[i];
        }

        SpMVL2 = sqrt(SpMVL2);
    }

    cout << "dot: " << dot << endl;
    
    cout << "axpby - sum: " << axpbySum << endl;
    cout << "axpby - L2: " << axpbyL2 << endl;

    cout << "SpMV - sum: " << SpMVSum << endl;
    cout << "SpMV - L2: " << SpMVL2 << endl;

    cout << "==== End ====" << endl;
}

int main()
{
    int Nx = 3;
    int Ny = 3;
    int Nz = 3;
    //int N = Nx * Ny * Nz;

    //int (*Col)[7] = new int[N][7];
    //double (*Val)[7] = new double[N][7];

    //generateMatrix(Nx, Ny, Nz, Col, Val);

    test(false, Nx, Ny, Nz);

    return 0;
}