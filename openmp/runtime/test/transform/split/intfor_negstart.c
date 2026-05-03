// RUN: %libomp-compile-and-run | FileCheck %s --match-full-lines

#include <stdlib.h>
#include <stdio.h>

int main() {
  int n = 8;
  printf("do\n");
#pragma omp split counts(1, omp_fill, 1)
  for (int i = -1; i <= n; ++i)
    printf("i=%d\n", i);
  printf("done\n");
  return EXIT_SUCCESS;
}

// CHECK:      do
// CHECK-NEXT: i=-1
// CHECK-NEXT: i=0
// CHECK-NEXT: i=1
// CHECK-NEXT: i=2
// CHECK-NEXT: i=3
// CHECK-NEXT: i=4
// CHECK-NEXT: i=5
// CHECK-NEXT: i=6
// CHECK-NEXT: i=7
// CHECK-NEXT: i=8
// CHECK-NEXT: done
