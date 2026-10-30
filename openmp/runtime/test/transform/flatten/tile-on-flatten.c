// RUN: %libomp-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <stdlib.h>
#include <stdio.h>

int main() {
  printf("do\n");
#pragma omp tile sizes(2)
#pragma omp flatten
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      printf("i=%d j=%d\n", i, j);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// Tile consumes the single loop produced by flatten. For this nest the
// tiled 1-D product still visits every (i, j) in row-major order.
// CHECK:      do
// CHECK-NEXT: i=0 j=0
// CHECK-NEXT: i=0 j=1
// CHECK-NEXT: i=1 j=0
// CHECK-NEXT: i=1 j=1
// CHECK-NEXT: done
