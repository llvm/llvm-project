// RUN: %libomp-cxx-compile-and-run | FileCheck %s --match-full-lines
// RUN: %libomp-cxx-compile -O2 && %libomp-run | FileCheck %s --match-full-lines

// collapse(3) reaches through a tiled loop to include the inner `j` loop.

#ifndef HEADER
#define HEADER

#include <cstdlib>
#include <cstdio>

int main() {
  printf("do\n");
#pragma omp parallel for collapse(3) num_threads(1)
#pragma omp tile sizes(4)
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 2; ++j)
      printf("i=%d j=%d\n", i, j);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      do
// CHECK-NEXT: i=0 j=0
// CHECK-NEXT: i=0 j=1
// CHECK-NEXT: i=1 j=0
// CHECK-NEXT: i=1 j=1
// CHECK-NEXT: i=2 j=0
// CHECK-NEXT: i=2 j=1
// CHECK-NEXT: i=3 j=0
// CHECK-NEXT: i=3 j=1
// CHECK-NEXT: i=4 j=0
// CHECK-NEXT: i=4 j=1
// CHECK-NEXT: i=5 j=0
// CHECK-NEXT: i=5 j=1
// CHECK-NEXT: done
