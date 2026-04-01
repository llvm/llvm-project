// RUN: %libomp-compile-and-run | FileCheck %s --match-full-lines

#include <stdlib.h>
#include <stdio.h>

int main() {
  int v = 3;
  printf("do\n");
#pragma omp split counts(v, omp_fill)
  for (int i = 0; i < 10; ++i)
    printf("i=%d\n", i);
  printf("done\n");
  return EXIT_SUCCESS;
}

// CHECK:      do
// CHECK-NEXT: i=0
// CHECK-NEXT: i=1
// CHECK-NEXT: i=2
// CHECK-NEXT: i=3
// CHECK-NEXT: i=4
// CHECK-NEXT: i=5
// CHECK-NEXT: i=6
// CHECK-NEXT: i=7
// CHECK-NEXT: i=8
// CHECK-NEXT: i=9
// CHECK-NEXT: done
