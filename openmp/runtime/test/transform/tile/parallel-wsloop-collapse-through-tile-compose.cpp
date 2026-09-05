// RUN: %libomp-cxx-compile-and-run | FileCheck %s --match-full-lines
// RUN: %libomp-cxx-compile -O2 && %libomp-run | FileCheck %s --match-full-lines

// collapse consuming a tile composed with reverse.

#ifndef HEADER
#define HEADER

#include <cstdlib>
#include <cstdio>

int main() {
  printf("rev2\n");
#pragma omp parallel for collapse(2) num_threads(1)
#pragma omp tile sizes(2)
#pragma omp reverse
  for (int i = 0; i < 5; ++i)
    printf("i=%d\n", i);

  printf("rev3\n");
#pragma omp parallel for collapse(3) num_threads(1)
#pragma omp tile sizes(2)
#pragma omp reverse
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 2; ++j)
      printf("i=%d j=%d\n", i, j);

  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      rev2
// CHECK-NEXT: i=4
// CHECK-NEXT: i=3
// CHECK-NEXT: i=2
// CHECK-NEXT: i=1
// CHECK-NEXT: i=0
// CHECK-NEXT: rev3
// CHECK-NEXT: i=4 j=0
// CHECK-NEXT: i=4 j=1
// CHECK-NEXT: i=3 j=0
// CHECK-NEXT: i=3 j=1
// CHECK-NEXT: i=2 j=0
// CHECK-NEXT: i=2 j=1
// CHECK-NEXT: i=1 j=0
// CHECK-NEXT: i=1 j=1
// CHECK-NEXT: i=0 j=0
// CHECK-NEXT: i=0 j=1
// CHECK-NEXT: done
