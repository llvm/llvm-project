#include <math.h>
#include <stdio.h>

#define TYPE double
#define PRINT_PRECISION_FORMAT "%0.15lf"
#define TOLERANCE 1e-17f
#define SQRT sqrt

__attribute__((noinline))
void solve_system(TYPE A11, TYPE A12, TYPE A13, TYPE A14,
                  TYPE A21, TYPE A22, TYPE A23, TYPE A24,
                  TYPE A31, TYPE A32, TYPE A33, TYPE A34,
                  TYPE A41, TYPE A42, TYPE A43, TYPE A44,
                  TYPE b1, TYPE b2, TYPE b3, TYPE b4) {
  // Tolerance
  TYPE eps = TOLERANCE;

  // Error
  TYPE e = 1.0f;

  // Initial solution
  TYPE x1 = 0.0f;
  TYPE x2 = 0.0f;
  TYPE x3 = 0.0f;
  TYPE x4 = 0.0f;

  // Jacobi iteration
  while (e > eps) {
    TYPE x_n1 = ((b1 / A11) - ((A12 / A11) * x2)) - ((A13 / A11) * x3) - ((A14 / A11) * x4);
    TYPE x_n2 = ((b2 / A22) - ((A21 / A22) * x1)) - ((A23 / A22) * x3) - ((A24 / A22) * x4);
    TYPE x_n3 = ((b3 / A33) - ((A31 / A33) * x1)) - ((A32 / A33) * x2) - ((A34 / A33) * x4);
    TYPE x_n4 = ((b4 / A44) - ((A41 / A44) * x1)) - ((A42 / A44) * x2) - ((A43 / A44) * x3);

    e = SQRT((x_n1-x1) * (x_n1-x1) +
             (x_n2-x2) * (x_n2-x2) +
             (x_n3-x3) * (x_n3-x3) +
             (x_n4-x4) * (x_n4-x4));

    x1 = x_n1;
    x2 = x_n2;
    x3 = x_n3;
    x4 = x_n4;
  }

  printf("x1 = "PRINT_PRECISION_FORMAT"\n", x1);
  printf("x2 = "PRINT_PRECISION_FORMAT"\n", x2);
  printf("x3 = "PRINT_PRECISION_FORMAT"\n", x3);
  printf("x4 = "PRINT_PRECISION_FORMAT"\n", x4);
}

int main() {
  // Matrix A
  TYPE A11 = 1.0, A12 = 0.1, A13 = 0.2, A14 = 0.3;
  TYPE A21 = 0.3, A22 = 1.0, A23 = 0.1, A24 = 0.2;
  TYPE A31 = 0.3, A32 = 0.1, A33 = 1.0, A34 = 0.1;
  TYPE A41 = 0.1, A42 = 0.2, A43 = 0.3, A44 = 1.0;

  // Vector b
  TYPE b1 = 1.0;
  TYPE b2 = 1.0;
  TYPE b3 = 1.0;
  TYPE b4 = 1.0;

  solve_system(A11, A12, A13, A14,
               A21, A22, A23, A24,
               A31, A32, A33, A34,
               A41, A42, A43, A44,
               b1, b2, b3, b4;

  return 0;
}