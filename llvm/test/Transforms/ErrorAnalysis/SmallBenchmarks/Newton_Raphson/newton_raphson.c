#include <stdio.h>
#include <math.h>

#define TYPE double
#define PRINT_PRECISION_FORMAT "%0.15lf"
#define FABS fabs

// Inputs
#define x 0.0
#define TOLERANCE 0.000001
#define n 1000

__attribute__((noinline))
TYPE newton_raphson(TYPE x_I) {
  TYPE x_I_1 = 0.0;
  TYPE e = 1.0;
  int I = 0;

  for (I = 0; (e > TOLERANCE) && (I < n); ++I) {
    // Compute powers of x_I to be used for Newtons method.
    TYPE pow_2 = x_I * x_I;
    TYPE pow_3 = pow_2 * x_I;
    TYPE pow_4 = pow_3 * x_I;
    TYPE pow_5 = pow_4 * x_I;
    TYPE pow_6 = pow_5 * x_I;
    TYPE pow_7 = pow_6 * x_I;

    // f = (x-2)^5 = x^5 - 10x^4 + 40x^3 - 80x^2 + 80x - 32
    TYPE f = ((((pow_5 - 10.0*pow_4) + 40.0*pow_3) - 80.0*pow_2) + 80.0*x_I) - 32.0;
    TYPE df = (((5.0*pow_4 - 40.0*pow_3) + 120.0*pow_2) - 160.0*x_I) + 80.0;
    x_I = x_I - f/df;
    e = FABS(x_I_1 - x_I);
    x_I_1 = x_I;

    printf("e = "PRINT_PRECISION_FORMAT"\n", e);
    printf("x = "PRINT_PRECISION_FORMAT"\n", x_I);
  }

  printf("Iterations = %d\n", I);

  return x_I;
}

int main() {
  printf("x = "PRINT_PRECISION_FORMAT"\n", newton_raphson(x));

  return 0;
}