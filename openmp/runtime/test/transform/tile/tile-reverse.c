// RUN: %libomp-compile -fopenmp-version=60 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

// tile restores the variables of a reverse it consumes, and leaves an inner
// one untouched when the enclosing loop has no iterations.

#ifndef HEADER
#define HEADER

#include <stdio.h>
#include <stdlib.h>

static void test_tile_reverse(void) {
  int i = -1;
#pragma omp tile sizes(2)
#pragma omp reverse
  for (i = 0; i < 5; i++)
    ;
  printf("tile_reverse: i=%d\n", i);
}

static void test_tile_nested_reverse(int n) {
  int i = 77, j = 111;
#pragma omp tile sizes(2, 3)
  for (i = 0; i < n; i++)
#pragma omp reverse
    for (j = 0; j < 5; j++)
      ;
  printf("tile_nested_reverse n=%d: i=%d j=%d\n", n, i, j);
}

int main(void) {
  test_tile_reverse();
  test_tile_nested_reverse(4);
  test_tile_nested_reverse(0);
  return EXIT_SUCCESS;
}
#endif

// CHECK:      tile_reverse: i=5
// CHECK-NEXT: tile_nested_reverse n=4: i=4 j=5
// CHECK-NEXT: tile_nested_reverse n=0: i=0 j=111
