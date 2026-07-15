// RUN: %libomp-cxx-compile-and-run | FileCheck %s --match-full-lines

// XFAIL: *
// collapse(3) through stacked tiles is not supported.

#ifndef HEADER
#define HEADER

#include <cstdlib>
#include <cstdio>

int main() {
  printf("do\n");
#pragma omp parallel for collapse(3) num_threads(1)
#pragma omp tile sizes(2)
#pragma omp tile sizes(4)
  for (int i = 0; i < 6; ++i)
    printf("i=%d\n", i);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      do
// CHECK-NEXT: i=0
// CHECK-NEXT: i=1
// CHECK-NEXT: i=2
// CHECK-NEXT: i=3
// CHECK-NEXT: i=4
// CHECK-NEXT: i=5
// CHECK-NEXT: done
