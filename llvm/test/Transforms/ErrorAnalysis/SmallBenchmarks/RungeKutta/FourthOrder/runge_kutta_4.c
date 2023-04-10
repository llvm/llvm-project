#include <stdio.h>
#include <math.h>

#define TYPE float
#define PRINT_PRECISION_FORMAT "%0.7f"

#define TOLERANCE 0.005
#define N 500

__attribute__((noinline))
void runge_kutta(TYPE h, TYPE y_n, TYPE c) {
  // Sixth
  TYPE sixth = 1.0 / 6.0;

  // Shape of curve
  TYPE k = 1.2;

  for (int I = 0; I < N; ++I) {

    // Calculate the k1, k2, k3, and k4
    TYPE v = c - y_n;
    TYPE k1 = (k * v) * v;

    TYPE v_1 = c - (y_n + ((0.5 * h) * k1));
    TYPE k2 = (k * v_1) * v_1;

    TYPE v_2 = c - (y_n + ((0.5 * h) * k2));
    TYPE k3 = (k * v_2) * v_2;

    TYPE v_3 = c - (y_n + (h * k3));
    TYPE k4 = (k * v_3) * v_3;

    // Calculate the new y_n
    y_n = y_n + ((sixth * h) * (((k1 + (2.0 * k2)) + (2.0 * k3)) + k4));

    printf("y_n: "PRINT_PRECISION_FORMAT"\n", y_n);
  }
}

int main() {
  TYPE h = 0.1;
  TYPE y_n = 0.0;
  TYPE c = 1.0;
  runge_kutta(h, y_n, c);

  return 0;
}
