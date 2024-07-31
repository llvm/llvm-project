// RUN: %libomp-compile-and-run

// Non-rectangular loop collapsing.
//
// Nested loops conform to OpenMP 5.2 standard,
// inner loops bounds may depend on outer loops induction variables.

#define LOOP_TYPES int
#define COMPARE0 <
#define COMPARE1 <=
#define COMPARE2 >=
#define LOOP                                                                   \
  for (i = iLB; i COMPARE0 iUB; i += iStep)                                    \
    for (j = jA0; j COMPARE1 jB0; j += jStep)                                  \
      for (k = kA0; k COMPARE2 kB0; k += kStep)
#include "collapse_test.inc"

int main() {
  int fail;

  iLB = -2;
  iUB = 3;
  jA0 = -3;
  jA1 = 0;
  jB0 = -6;
  jB1 = 0;
  kA0 = -2;
  kA1 = 0;
  kB0 = -4;
  kB1 = 0;
  iStep = -1;
  jStep = -1;
  kStep = -4;
  PRINTF("\nOne off iLB=%d; iUB=%d; jA0=%d; jA1=%d; jB0=%d; jB1=%d; kA0=%d; "
         "kA1=%d; kB0=%d; kB1=%d; iStep=%d; jStep=%d; kStep=%d;\n",
         iLB, iUB, jA0, jA1, jB0, jB1, kA0, kA1, kB0, kB1, iStep, jStep, kStep);
  fail = (test() == 0);

  if (!fail) {

    for (iStep = 2; iStep <= 6; iStep += 2) {
      for (jA0 = -6; jA0 <= 6; jA0 += 3) {
        for (jB0 = -3; jB0 <= 10; jB0 += 3) {
          for (jStep = 1; jStep <= 10; jStep += 2) {
            for (kA0 = -2; kA0 <= 4; ++kA0) {
              for (kB0 = -4; kB0 <= 2; ++kB0) {
                for (kStep = -2; kStep >= -10; kStep -= 4) {
                  {
                    PRINTF("\nTrying iLB=%d; iUB=%d; jA0=%d; jA1=%d; jB0=%d; "
                           "jB1=%d; kA0=%d; kA1=%d; kB0=%d; kB1=%d; iStep=%d; "
                           "jStep=%d; kStep=%d;\n",
                           iLB, iUB, jA0, jA1, jB0, jB1, kA0, kA1, kB0, kB1,
                           iStep, jStep, kStep);
                    fail = fail || (test() == 0);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return fail;
}
