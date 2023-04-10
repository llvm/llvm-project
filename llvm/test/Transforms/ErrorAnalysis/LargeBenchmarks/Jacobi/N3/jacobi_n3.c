#include <math.h>
#include <stdio.h>

#define TYPE double
#define PRINT_PRECISION_FORMAT "%0.15lf"
#define TOLERANCE 1e-17f
#define SQRT sqrt

__attribute__((noinline))
void solve_system(TYPE A11, TYPE A12, TYPE A13,
                  TYPE A21, TYPE A22, TYPE A23,
                  TYPE A31, TYPE A32, TYPE A33,
                  TYPE b1, TYPE b2, TYPE b3) {
  // Tolerance
  TYPE eps = TOLERANCE;

  // Error
  TYPE e = 1.0f;

  // Initial solution
  TYPE x1 = 0.0f;
  TYPE x2 = 0.0f;
  TYPE x3 = 0.0f;

  // Jacobi iteration
  while (e > eps) {
    TYPE x_n1 = ((b1 / A11) - ((A12 / A11) * x2)) - ((A13 / A11) * x3);
    TYPE x_n2 = ((b2 / A22) - ((A21 / A22) * x1)) - ((A23 / A22) * x3);
    TYPE x_n3 = ((b3 / A33) - ((A31 / A33) * x1)) - ((A32 / A33) * x2);

    e = SQRT((x_n1-x1) * (x_n1-x1) +
             (x_n2-x2) * (x_n2-x2) +
             (x_n3-x3) * (x_n3-x3));

    x1 = x_n1;
    x2 = x_n2;
    x3 = x_n3;
  }

  printf("x1 = "PRINT_PRECISION_FORMAT"\n", x1);
  printf("x2 = "PRINT_PRECISION_FORMAT"\n", x2);
  printf("x3 = "PRINT_PRECISION_FORMAT"\n", x3);
}

int main() {
  // Matrix A
  TYPE A11 = 1.0, A12 = 0.1, A13 = 0.2;
  TYPE A21 = 0.3, A22 = 1.0, A23 = 0.1;
  TYPE A31 = 0.3, A32 = 0.1, A33 = 1.0;

  // Vector b
  TYPE b1 = 1.0;
  TYPE b2 = 1.0;
  TYPE b3 = 1.0;

  solve_system(A11, A12, A13,
               A21, A22, A23,
               A31, A32, A33,
               b1, b2, b3);

  return 0;
}