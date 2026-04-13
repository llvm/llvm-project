// RUN: %libomp-compile-and-run | FileCheck %s --match-full-lines

#include <stdlib.h>
#include <stdio.h>

int main() {
  int n = 19;
  int c = 3;
  printf("do\n");
#pragma omp split counts(1, omp_fill, 1)
  for (int i = 7; i < n; i += c)
    printf("i=%d\n", i);
  printf("done\n");
  return EXIT_SUCCESS;
}

// CHECK:      do
// CHECK-NEXT: i=7
// CHECK-NEXT: i=10
// CHECK-NEXT: i=13
// CHECK-NEXT: i=16
// CHECK-NEXT: done
