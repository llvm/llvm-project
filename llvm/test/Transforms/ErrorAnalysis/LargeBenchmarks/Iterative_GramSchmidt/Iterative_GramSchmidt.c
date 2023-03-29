#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define TYPE float
#define PRINT_PRECISION_FORMAT "%0.7f"

TYPE ex0(TYPE Q11, TYPE Q12, TYPE Q13, TYPE Q21, TYPE Q22, TYPE Q23, TYPE Q31, TYPE Q32, TYPE Q33) {
  TYPE eps = 5e-06f;
  TYPE h1 = 0.0f;
  TYPE h2 = 0.0f;
  TYPE h3 = 0.0f;
  TYPE qj1 = Q31;
  TYPE qj2 = Q32;
  TYPE qj3 = Q33;
  TYPE r1 = 0.0f;
  TYPE r2 = 0.0f;
  TYPE r3 = 0.0f;
  TYPE r = ((qj1 * qj1) + (qj2 * qj2)) + (qj3 * qj3);
  TYPE rjj = 0.0f;
  TYPE e = 10.0f;
  TYPE i = 1.0f;
  TYPE rold = sqrtf(r);
  int tmp = e > eps;

  while (tmp) {
          h1 = ((Q11 * qj1) + (Q21 * qj2)) + (Q31 * qj3);
          h2 = ((Q12 * qj1) + (Q22 * qj2)) + (Q32 * qj3);
          h3 = ((Q13 * qj1) + (Q23 * qj2)) + (Q33 * qj3);
          qj1 = qj1 - (((Q11 * h1) + (Q12 * h2)) + (Q13 * h3));
          qj2 = qj2 - (((Q21 * h1) + (Q22 * h2)) + (Q23 * h3));
          qj3 = qj3 - (((Q31 * h1) + (Q32 * h2)) + (Q33 * h3));
          r1 = r1 + h1;
          r2 = r2 + h2;
          r3 = r3 + h3;
          r = ((qj1 * qj1) + (qj2 * qj2)) + (qj3 * qj3);
          rjj = sqrtf(r);
          e = fabsf((1.0f - (rjj / rold)));
          i = i + 1.0f;
          rold = rjj;
          tmp = e > eps;
  }

  return qj1;
}

int main() {
  TYPE Q11 = 1.0;
  TYPE Q12 = 0.5;
  TYPE Q13 = 0.5;
  TYPE Q21 = 0.1;
  TYPE Q22 = 1.0;
  TYPE Q23 = 0.5;
  TYPE Q31 = 1/2592;
  TYPE Q32 = 1/2601;
  TYPE Q33 = 1/2583;
  TYPE res;

  res = ex0(Q11, Q12, Q13, Q21, Q22, Q23, Q31, Q32, Q33);

  printf("Result = "PRINT_PRECISION_FORMAT"\n", res);

  return 0;
}