// RUN: %libomp-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  int n = 2;
  printf("do\n");
#pragma omp flatten depth(2)
  for (size_t i = 0; i < (size_t)n; ++i)
    for (int j = 0; j < n; ++j)
      printf("i=%zu j=%d\n", i, j);
  printf("done\n");
  return EXIT_SUCCESS;
}

// CHECK:      do
// CHECK-NEXT: i=0 j=0
// CHECK-NEXT: i=0 j=1
// CHECK-NEXT: i=1 j=0
// CHECK-NEXT: i=1 j=1
// CHECK-NEXT: done
