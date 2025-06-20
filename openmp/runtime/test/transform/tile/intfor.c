// RUN: %libomp-compile-and-run | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <stdlib.h>
#include <stdio.h>

// TODO: The OpenMP specification explicitly does not define when and how often
// expressions in the clause are evaluated. Currently Clang evaluates it again
// whenever needed, but function calls in clauses are not common. A better
// implementation would evaluate it just once and reuse the result.
static int tilesize(int i) {
  printf("tilesize(%d)\n", i);
  return 3;
}

int main() {
  printf("do\n");
#pragma omp tile sizes(tilesize(1), tilesize(2))
  for (int i = 7; i < 19; i += 3)
    for (int j = 7; j < 20; j += 3)
      printf("i=%d j=%d\n", i, j);
  printf("done\n");
  return EXIT_SUCCESS;
}

#endif /* HEADER */

// CHECK:      do
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=7 j=7
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=7 j=10
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=7 j=13
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=10 j=7
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=10 j=10
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=10 j=13
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=13 j=7
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=13 j=10
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=13 j=13
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=7 j=16
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=7 j=19
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=10 j=16
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=10 j=19
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=13 j=16
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=13 j=19
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=16 j=7
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=16 j=10
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=16 j=13
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=16 j=16
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: i=16 j=19
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(2)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: tilesize(1)
// CHECK-NEXT: done