// RUN: %libomp-cxx-compile-and-run | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <cstdlib>
#include <cstdio>

int main() {
  printf("do\n");
#pragma omp parallel for collapse(3) num_threads(1)
  for (int i = 0; i < 3; ++i)
#pragma omp tile sizes(3, 3)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 5; ++k)
        printf("i=%d j=%d k=%d\n", i, j, k);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      do

// Full tile
// CHECK-NEXT: i=0 j=0 k=0
// CHECK-NEXT: i=0 j=0 k=1
// CHECK-NEXT: i=0 j=0 k=2
// CHECK-NEXT: i=0 j=1 k=0
// CHECK-NEXT: i=0 j=1 k=1
// CHECK-NEXT: i=0 j=1 k=2
// CHECK-NEXT: i=0 j=2 k=0
// CHECK-NEXT: i=0 j=2 k=1
// CHECK-NEXT: i=0 j=2 k=2

// Partial tile
// CHECK-NEXT: i=0 j=0 k=3
// CHECK-NEXT: i=0 j=0 k=4
// CHECK-NEXT: i=0 j=1 k=3
// CHECK-NEXT: i=0 j=1 k=4
// CHECK-NEXT: i=0 j=2 k=3
// CHECK-NEXT: i=0 j=2 k=4

// Partial tile
// CHECK-NEXT: i=0 j=3 k=0
// CHECK-NEXT: i=0 j=3 k=1
// CHECK-NEXT: i=0 j=3 k=2

// Partial tile
// CHECK-NEXT: i=0 j=3 k=3
// CHECK-NEXT: i=0 j=3 k=4

// Full tile
// CHECK-NEXT: i=1 j=0 k=0
// CHECK-NEXT: i=1 j=0 k=1
// CHECK-NEXT: i=1 j=0 k=2
// CHECK-NEXT: i=1 j=1 k=0
// CHECK-NEXT: i=1 j=1 k=1
// CHECK-NEXT: i=1 j=1 k=2
// CHECK-NEXT: i=1 j=2 k=0
// CHECK-NEXT: i=1 j=2 k=1
// CHECK-NEXT: i=1 j=2 k=2

// Partial tiles
// CHECK-NEXT: i=1 j=0 k=3
// CHECK-NEXT: i=1 j=0 k=4
// CHECK-NEXT: i=1 j=1 k=3
// CHECK-NEXT: i=1 j=1 k=4
// CHECK-NEXT: i=1 j=2 k=3
// CHECK-NEXT: i=1 j=2 k=4
// CHECK-NEXT: i=1 j=3 k=0
// CHECK-NEXT: i=1 j=3 k=1
// CHECK-NEXT: i=1 j=3 k=2
// CHECK-NEXT: i=1 j=3 k=3
// CHECK-NEXT: i=1 j=3 k=4

// Full tile
// CHECK-NEXT: i=2 j=0 k=0
// CHECK-NEXT: i=2 j=0 k=1
// CHECK-NEXT: i=2 j=0 k=2
// CHECK-NEXT: i=2 j=1 k=0
// CHECK-NEXT: i=2 j=1 k=1
// CHECK-NEXT: i=2 j=1 k=2
// CHECK-NEXT: i=2 j=2 k=0
// CHECK-NEXT: i=2 j=2 k=1
// CHECK-NEXT: i=2 j=2 k=2

// Partial tiles
// CHECK-NEXT: i=2 j=0 k=3
// CHECK-NEXT: i=2 j=0 k=4
// CHECK-NEXT: i=2 j=1 k=3
// CHECK-NEXT: i=2 j=1 k=4
// CHECK-NEXT: i=2 j=2 k=3
// CHECK-NEXT: i=2 j=2 k=4
// CHECK-NEXT: i=2 j=3 k=0
// CHECK-NEXT: i=2 j=3 k=1
// CHECK-NEXT: i=2 j=3 k=2
// CHECK-NEXT: i=2 j=3 k=3
// CHECK-NEXT: i=2 j=3 k=4
// CHECK-NEXT: done
