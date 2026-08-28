// RUN: %libomp-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

// 'depth(3)' fully flattens a three-deep perfectly nested loop into one loop.
// This verifies that the flattened loop preserves the exact row-major
// visitation order of the original nest, including non-unit steps and non-zero
// start values.

#ifndef HEADER
#define HEADER

#include <stdlib.h>
#include <stdio.h>

int main() {
  printf("do\n");
#pragma omp flatten depth(3)
  for (int i = 7; i < 11; i += 2)
    for (int j = 1; j < 4; j += 1)
      for (int k = 0; k < 2; ++k)
        printf("i=%d j=%d k=%d\n", i, j, k);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      do
// CHECK-NEXT: i=7 j=1 k=0
// CHECK-NEXT: i=7 j=1 k=1
// CHECK-NEXT: i=7 j=2 k=0
// CHECK-NEXT: i=7 j=2 k=1
// CHECK-NEXT: i=7 j=3 k=0
// CHECK-NEXT: i=7 j=3 k=1
// CHECK-NEXT: i=9 j=1 k=0
// CHECK-NEXT: i=9 j=1 k=1
// CHECK-NEXT: i=9 j=2 k=0
// CHECK-NEXT: i=9 j=2 k=1
// CHECK-NEXT: i=9 j=3 k=0
// CHECK-NEXT: i=9 j=3 k=1
// CHECK-NEXT: done
