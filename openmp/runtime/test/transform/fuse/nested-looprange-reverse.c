// RUN: %libomp-compile -fopenmp-version=60 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

// Nested fuse where the inner looprange fuse leaves a reversed loop out of
// range. After the construct, each loop variable must hold its loop-exit
// value (OpenMP 6.0).

#ifndef HEADER
#define HEADER

#include <stdlib.h>
#include <stdio.h>

int main() {
  int i = -1, j = -1, k = -1, l = -1;
#pragma omp fuse
  {
#pragma omp fuse looprange(1, 2)
    {
      for (i = 0; i < 10; i++)
        ;
      for (j = 0; j < 15; j++)
        ;
#pragma omp reverse
      for (k = 0; k < 5; k++)
        ;
    }
    for (l = 0; l < 4; l++)
      ;
  }
  printf("i=%d j=%d k=%d l=%d\n", i, j, k, l);
  return EXIT_SUCCESS;
}
#endif /* HEADER */

// CHECK: i=10 j=15 k=5 l=4
