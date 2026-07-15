// RUN: %libomp-cxx-compile-and-run | FileCheck %s --match-full-lines
// RUN: %libomp-cxx-compile -O2 && %libomp-run | FileCheck %s --match-full-lines

// 2-D tile consumed by collapse(4) (full) and collapse(3) (partial).

#ifndef HEADER
#define HEADER

#include <cstdlib>
#include <cstdio>

int main() {
  printf("full4\n");
#pragma omp parallel for collapse(4) num_threads(1)
#pragma omp tile sizes(2, 2)
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      printf("i=%d j=%d\n", i, j);

  printf("part3\n");
#pragma omp parallel for collapse(3) num_threads(1)
#pragma omp tile sizes(2, 2)
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      printf("i=%d j=%d\n", i, j);

  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      full4
// CHECK-NEXT: i=0 j=0
// CHECK-NEXT: i=0 j=1
// CHECK-NEXT: i=1 j=0
// CHECK-NEXT: i=1 j=1
// CHECK-NEXT: i=0 j=2
// CHECK-NEXT: i=1 j=2
// CHECK-NEXT: i=2 j=0
// CHECK-NEXT: i=2 j=1
// CHECK-NEXT: i=2 j=2
// CHECK-NEXT: part3
// CHECK-NEXT: i=0 j=0
// CHECK-NEXT: i=0 j=1
// CHECK-NEXT: i=1 j=0
// CHECK-NEXT: i=1 j=1
// CHECK-NEXT: i=0 j=2
// CHECK-NEXT: i=0 j=3
// CHECK-NEXT: i=1 j=2
// CHECK-NEXT: i=1 j=3
// CHECK-NEXT: i=2 j=0
// CHECK-NEXT: i=2 j=1
// CHECK-NEXT: i=2 j=2
// CHECK-NEXT: i=2 j=3
// CHECK-NEXT: done
