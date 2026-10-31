// RUN: %libomp-compile -fopenmp-version=60 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

// fuse restores the variable of a partially unrolled loop it consumes.

#ifndef HEADER
#define HEADER

#include <stdio.h>
#include <stdlib.h>

int main(void) {
  int i = -1, j = -1;
#pragma omp fuse
  {
#pragma omp unroll partial(2)
    for (i = 0; i < 5; i++)
      ;
    for (j = 0; j < 5; j++)
      ;
  }
  printf("i=%d j=%d\n", i, j);
  return EXIT_SUCCESS;
}
#endif

// CHECK: i=5 j=5
