// RUN: %libomp-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <stdlib.h>
#include <stdio.h>

int main() {
  printf("do\n");
#pragma omp flatten
  for (int i = 5; i < 11; i += 2)
    for (int j = 1; j < 7; j += 2)
      printf("i=%d j=%d\n", i, j);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// The flattened loop visits the original iteration space in the original
// (row-major) order: for each outer iteration the inner loop runs fully.
// CHECK:      do
// CHECK-NEXT: i=5 j=1
// CHECK-NEXT: i=5 j=3
// CHECK-NEXT: i=5 j=5
// CHECK-NEXT: i=7 j=1
// CHECK-NEXT: i=7 j=3
// CHECK-NEXT: i=7 j=5
// CHECK-NEXT: i=9 j=1
// CHECK-NEXT: i=9 j=3
// CHECK-NEXT: i=9 j=5
// CHECK-NEXT: done
