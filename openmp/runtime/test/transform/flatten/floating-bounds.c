// RUN: %libomp-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <stdlib.h>
#include <stdio.h>

int main(void) {
  int count = 0;

#pragma omp flatten
  for (int i = 0; i < 2; ++i)
    for (int j = 0.; j < 2; ++j)
      ++count;
  printf("floating-init=%d\n", count);

  count = 0;
#pragma omp flatten
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2.; ++j)
      ++count;
  printf("floating-bound=%d\n", count);

  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      floating-init=4
// CHECK-NEXT: floating-bound=4
