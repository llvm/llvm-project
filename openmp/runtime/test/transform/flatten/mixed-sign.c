// RUN: %libomp-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

#include <stdio.h>
#include <stdlib.h>

typedef __SIZE_TYPE__ size_t;

static void flatten(int n) {
#pragma omp flatten depth(2)
  for (size_t i = 0; i < (size_t)n; ++i)
    for (int j = 0; j < n; ++j)
      printf("i=%zu j=%d\n", i, j);
}

int main(void) {
  printf("n2-begin\n");
  flatten(2);
  printf("n2-end\n");

  printf("n0-begin\n");
  flatten(0);
  printf("n0-end\n");

  printf("nneg-begin\n");
  flatten(-2);
  printf("nneg-end\n");
  return EXIT_SUCCESS;
}

// CHECK:      n2-begin
// CHECK-NEXT: i=0 j=0
// CHECK-NEXT: i=0 j=1
// CHECK-NEXT: i=1 j=0
// CHECK-NEXT: i=1 j=1
// CHECK-NEXT: n2-end
// CHECK-NEXT: n0-begin
// CHECK-NEXT: n0-end
// CHECK-NEXT: nneg-begin
// CHECK-NEXT: nneg-end
