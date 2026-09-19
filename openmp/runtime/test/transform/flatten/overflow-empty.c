// RUN: %libomp-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static int Count;

static void flatten(int Lower, int Upper) {
#pragma omp flatten
  for (int I = Lower; I < Upper; ++I)
    for (int J = 0; J < 2; ++J)
      ++Count;
}

int main(void) {
  // The trip-count expression may overflow or wrap for these bounds. The
  // original loop condition is false, so flatten must not evaluate that count
  // and must execute no iterations.
  flatten(INT_MAX, INT_MIN);
  printf("count=%d\n", Count);
  return EXIT_SUCCESS;
}

// CHECK: count=0
