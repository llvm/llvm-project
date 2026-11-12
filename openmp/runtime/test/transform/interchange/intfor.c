// RUN: %libomp-compile-and-run | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <stdlib.h>
#include <stdio.h>

int main() {
  printf("do\n");
#pragma omp interchange
  for (int i = 7; i < 17; i += 3)
    for (int j = 8; j < 18; j += 3)
      printf("i=%d j=%d\n", i, j);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      do
// CHECK-NEXT: i=7 j=8
// CHECK-NEXT: i=10 j=8
// CHECK-NEXT: i=13 j=8
// CHECK-NEXT: i=16 j=8
// CHECK-NEXT: i=7 j=11
// CHECK-NEXT: i=10 j=11
// CHECK-NEXT: i=13 j=11
// CHECK-NEXT: i=16 j=11
// CHECK-NEXT: i=7 j=14
// CHECK-NEXT: i=10 j=14
// CHECK-NEXT: i=13 j=14
// CHECK-NEXT: i=16 j=14
// CHECK-NEXT: i=7 j=17
// CHECK-NEXT: i=10 j=17
// CHECK-NEXT: i=13 j=17
// CHECK-NEXT: i=16 j=17
// CHECK-NEXT: done
