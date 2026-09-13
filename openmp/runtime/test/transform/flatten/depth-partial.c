// RUN: %libomp-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

// A depth(2) clause on a three-deep loop nest flattens only the outermost two
// loops; the innermost loop is left untouched and executes normally inside the
// flattened loop body.

#ifndef HEADER
#define HEADER

#include <stdlib.h>
#include <stdio.h>

int main() {
  printf("do\n");
#pragma omp flatten depth(2)
  for (int i = 5; i < 11; i += 2)
    for (int j = 1; j < 7; j += 2)
      for (int k = 0; k < 2; ++k)
        printf("i=%d j=%d k=%d\n", i, j, k);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      do
// CHECK-NEXT: i=5 j=1 k=0
// CHECK-NEXT: i=5 j=1 k=1
// CHECK-NEXT: i=5 j=3 k=0
// CHECK-NEXT: i=5 j=3 k=1
// CHECK-NEXT: i=5 j=5 k=0
// CHECK-NEXT: i=5 j=5 k=1
// CHECK-NEXT: i=7 j=1 k=0
// CHECK-NEXT: i=7 j=1 k=1
// CHECK-NEXT: i=7 j=3 k=0
// CHECK-NEXT: i=7 j=3 k=1
// CHECK-NEXT: i=7 j=5 k=0
// CHECK-NEXT: i=7 j=5 k=1
// CHECK-NEXT: i=9 j=1 k=0
// CHECK-NEXT: i=9 j=1 k=1
// CHECK-NEXT: i=9 j=3 k=0
// CHECK-NEXT: i=9 j=3 k=1
// CHECK-NEXT: i=9 j=5 k=0
// CHECK-NEXT: i=9 j=5 k=1
// CHECK-NEXT: done
