// RUN: %libomp-compile-and-run

// Non-rectangular loop collapsing.
//
// Nested loops conform to OpenMP 5.2 standard,
// inner loops bounds may depend on outer loops induction variables.

#define LOOP_TYPES int
#define LOOP                                                                   \
  for (i = iLB; i <= iUB; i += iStep)                                          \
    for (j = i + jA0; j <= i + jB0; j += jStep)                                \
      for (k = j + kA0; k <= j + kB0; k += kStep)

#include "collapse_test.inc"

int main() {
  int fail;
  iLB = -2;
  iUB = 3;
  jA0 = -7;
  jB0 = 13;
  kA0 = -20;
  kB0 = 111;
  iStep = 5;
  jStep = 9;
  kStep = 10;
  PRINTF("\nOne off iLB=%d; iUB=%d; jA0=%d; jB0=%d; kA0=%d; kB0=%d; iStep=%d; "
         "jStep=%d; kStep=%d;\n",
         iLB, iUB, jA0, jB0, kA0, kB0, iStep, jStep, kStep);
  fail = (test() == 0);
  return fail;
}
