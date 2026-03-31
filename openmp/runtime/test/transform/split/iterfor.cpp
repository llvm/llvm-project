// RUN: %libomp-cxx-compile-and-run | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <cstdlib>
#include <cstdio>

int main() {
  printf("do\n");
#pragma omp split counts(1, omp_fill, 1)
  for (int i = 7; i < 19; i += 3)
    printf("i=%d\n", i);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      do
// CHECK-NEXT: i=7
// CHECK-NEXT: i=10
// CHECK-NEXT: i=13
// CHECK-NEXT: i=16
// CHECK-NEXT: done
