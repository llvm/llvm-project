// RUN: %libomp-compile-and-run | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <stdlib.h>
#include <stdio.h>

int main() {
  printf("do\n");
#pragma omp reverse
  for (int i = 7; i < 19; i += 3)
    printf("i=%d\n", i);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      do
// CHECK-NEXT: i=16
// CHECK-NEXT: i=13
// CHECK-NEXT: i=10
// CHECK-NEXT: i=7
// CHECK-NEXT: done
