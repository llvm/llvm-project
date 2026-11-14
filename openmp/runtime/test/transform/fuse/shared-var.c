// RUN: %libomp-compile -fopenmp-version=60 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

// Loops of a fused sequence share one iteration variable. After the construct
// the variable must hold the loop-exit value of the last loop of the sequence,
// as without the directive, also when that loop is a consumed transformation.

#ifndef HEADER
#define HEADER

#include <stdio.h>
#include <stdlib.h>

static void test_reverse_last(void) {
  int i = -1;
#pragma omp fuse
  {
    for (i = 0; i < 4; i++)
      ;
#pragma omp reverse
    for (i = 0; i < 6; i++)
      ;
  }
  printf("reverse_last: i=%d\n", i);
}

static void test_unroll_last(void) {
  int i = -1;
#pragma omp fuse
  {
    for (i = 0; i < 4; i++)
      ;
#pragma omp unroll partial(2)
    for (i = 0; i < 6; i++)
      ;
  }
  printf("unroll_last: i=%d\n", i);
}

int main(void) {
  test_reverse_last();
  test_unroll_last();
  return EXIT_SUCCESS;
}
#endif

// CHECK:      reverse_last: i=6
// CHECK-NEXT: unroll_last: i=6
