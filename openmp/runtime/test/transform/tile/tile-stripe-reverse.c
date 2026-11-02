// RUN: %libomp-compile -fopenmp-version=60 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

// tile/stripe over a reverse that is not the innermost affected loop must
// keep the reversal (regression test for #222401).

#ifndef HEADER
#define HEADER

#include <stdio.h>
#include <stdlib.h>

int main() {
  printf("tile\n");
#pragma omp tile sizes(2, 2)
#pragma omp reverse
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      printf("i=%d j=%d\n", i, j);

  printf("stripe\n");
#pragma omp stripe sizes(2, 2)
#pragma omp reverse
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      printf("i=%d j=%d\n", i, j);

  printf("nested\n");
#pragma omp tile sizes(2, 2)
#pragma omp reverse
  for (int j = 0; j < 4; ++j) {
#pragma omp reverse
    for (int i = 0; i < 4; ++i)
      printf("j=%d i=%d\n", j, i);
  }

  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      tile
// CHECK-NEXT: i=3 j=0
// CHECK-NEXT: i=3 j=1
// CHECK-NEXT: i=2 j=0
// CHECK-NEXT: i=2 j=1
// CHECK-NEXT: i=3 j=2
// CHECK-NEXT: i=3 j=3
// CHECK-NEXT: i=2 j=2
// CHECK-NEXT: i=2 j=3
// CHECK-NEXT: i=1 j=0
// CHECK-NEXT: i=1 j=1
// CHECK-NEXT: i=0 j=0
// CHECK-NEXT: i=0 j=1
// CHECK-NEXT: i=1 j=2
// CHECK-NEXT: i=1 j=3
// CHECK-NEXT: i=0 j=2
// CHECK-NEXT: i=0 j=3
// CHECK-NEXT: stripe
// CHECK-NEXT: i=3 j=0
// CHECK-NEXT: i=3 j=1
// CHECK-NEXT: i=2 j=0
// CHECK-NEXT: i=2 j=1
// CHECK-NEXT: i=3 j=2
// CHECK-NEXT: i=3 j=3
// CHECK-NEXT: i=2 j=2
// CHECK-NEXT: i=2 j=3
// CHECK-NEXT: i=1 j=0
// CHECK-NEXT: i=1 j=1
// CHECK-NEXT: i=0 j=0
// CHECK-NEXT: i=0 j=1
// CHECK-NEXT: i=1 j=2
// CHECK-NEXT: i=1 j=3
// CHECK-NEXT: i=0 j=2
// CHECK-NEXT: i=0 j=3
// CHECK-NEXT: nested
// CHECK-NEXT: j=3 i=3
// CHECK-NEXT: j=3 i=2
// CHECK-NEXT: j=2 i=3
// CHECK-NEXT: j=2 i=2
// CHECK-NEXT: j=3 i=1
// CHECK-NEXT: j=3 i=0
// CHECK-NEXT: j=2 i=1
// CHECK-NEXT: j=2 i=0
// CHECK-NEXT: j=1 i=3
// CHECK-NEXT: j=1 i=2
// CHECK-NEXT: j=0 i=3
// CHECK-NEXT: j=0 i=2
// CHECK-NEXT: j=1 i=1
// CHECK-NEXT: j=1 i=0
// CHECK-NEXT: j=0 i=1
// CHECK-NEXT: j=0 i=0
// CHECK-NEXT: done
