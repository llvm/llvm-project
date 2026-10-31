// RUN: %libomp-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

// A loop transformation consumed by flatten: the variable of the transformed
// loop must hold its loop-exit value after the construct, as without the
// directive.

#ifndef HEADER
#define HEADER

#include <stdio.h>
#include <stdlib.h>

int main(void) {
  int i = -1, j = -1;
#pragma omp flatten
  for (i = 0; i < 4; i++)
#pragma omp reverse
    for (j = 0; j < 5; j++)
      ;
  printf("flatten_reverse_inner: i=%d j=%d\n", i, j);
  return EXIT_SUCCESS;
}
#endif

// CHECK: flatten_reverse_inner: i=4 j=5
