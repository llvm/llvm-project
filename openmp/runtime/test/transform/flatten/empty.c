// RUN: %libomp-compile-and-run | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <stdlib.h>
#include <stdio.h>

// The bounds are runtime values so that an empty iteration space does not
// trigger a compile-time division-by-zero diagnostic for the (never executed)
// induction-variable recovery.
static void flatten(int n, int m) {
#pragma omp flatten
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      printf("i=%d j=%d\n", i, j);
}

int main() {
  printf("empty-inner-begin\n");
  flatten(3, 0);
  printf("empty-inner-end\n");

  printf("empty-outer-begin\n");
  flatten(0, 3);
  printf("empty-outer-end\n");

  printf("single-begin\n");
  flatten(1, 1);
  printf("single-end\n");

  printf("neg-outer-begin\n");
#pragma omp flatten
  for (int i = 0; i < -1; ++i)
    for (int j = 0; j < 3; ++j)
      printf("i=%d j=%d\n", i, j);
  printf("neg-outer-end\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      empty-inner-begin
// CHECK-NEXT: empty-inner-end
// CHECK-NEXT: empty-outer-begin
// CHECK-NEXT: empty-outer-end
// CHECK-NEXT: single-begin
// CHECK-NEXT: i=0 j=0
// CHECK-NEXT: single-end
// CHECK-NEXT: neg-outer-begin
// CHECK-NEXT: neg-outer-end
