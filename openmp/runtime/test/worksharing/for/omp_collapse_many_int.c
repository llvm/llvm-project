// RUN: %libomp-compile-and-run
// XFAIL: true

// Non-rectangular loop collapsing.
//
// Nested loops conform to OpenMP 5.2 standard,
// inner loops bounds may depend on outer loops induction variables.

#define LOOP_TYPES int
#define LOOP                                                                   \
  for (i = iLB; i <= iUB; i += iStep)                                          \
    for (j = i * jA1 + jA0; j <= i * jB1 + jB0; j += jStep)                    \
      for (k = j * kA1 + kA0; k <= j * kB1 + kB0; k += kStep)
#include "collapse_test.inc"

int main() {
  int fail = 0;

  iLB = -2;
  iUB = 3;
  jA0 = -7;
  jA1 = -1;
  jB0 = 13;
  jB1 = 3;
  kA0 = -20;
  kA1 = -2;
  kB0 = 111;
  kB1 = -1;
  iStep = 5;
  jStep = 9;
  kStep = 10;
  PRINTF("\nOne off iLB=%d; iUB=%d; jA0=%d; jA1=%d; jB0=%d; jB1=%d; kA0=%d; "
         "kA1=%d; kB0=%d; kB1=%d; iStep=%d; jStep=%d; kStep=%d;\n",
         iLB, iUB, jA0, jA1, jB0, jB1, kA0, kA1, kB0, kB1, iStep, jStep, kStep);
  fail = fail || (test() == 0);

  if (!fail) {

    // NOTE: if a loop on some level won't execute  for all iterations of an
    // outer loop, it still should work. Runtime doesn't require lower bounds to
    // be <= upper bounds for all possible i, j, k.

    iLB = -2;
    iUB = 3;
    jA0 = -7;
    jB0 = 5;
    kA0 = -13;
    kB0 = 37;

    for (kA1 = -2; kA1 <= 2; ++kA1) { // <=
      for (kB1 = -2; kB1 <= 2; ++kB1) {
        for (jA1 = -3; jA1 <= 3; ++jA1) {
          for (jB1 = -3; jB1 <= 3; ++jB1) {
            for (iStep = 1; iStep <= 3; ++iStep) {
              for (jStep = 2; jStep <= 6; jStep += 2) {
                for (kStep = 2; kStep <= 8; kStep += 3) {
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

  return fail;
}
