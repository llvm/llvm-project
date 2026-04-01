// RUN: %libomp-compile-and-run | FileCheck %s --match-full-lines

// Tiny trip counts: trip=1 with counts(1, omp_fill) and trip=0.

#include <stdlib.h>
#include <stdio.h>

int main() {
  int n;

  n = 1;
  printf("trip1\n");
#pragma omp split counts(1, omp_fill)
  for (int i = 0; i < n; ++i)
    printf("i=%d\n", i);
  printf("end1\n");

  n = 0;
  printf("trip0\n");
#pragma omp split counts(omp_fill)
  for (int i = 0; i < n; ++i)
    printf("i=%d\n", i);
  printf("end0\n");

  return EXIT_SUCCESS;
}

// CHECK:      trip1
// CHECK-NEXT: i=0
// CHECK-NEXT: end1
// CHECK-NEXT: trip0
// CHECK-NEXT: end0
