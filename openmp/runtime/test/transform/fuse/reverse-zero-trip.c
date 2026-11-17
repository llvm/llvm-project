// RUN: %libomp-compile -fopenmp-version=60 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

// A reverse consumed by fuse keeps its pre-inits: its loop init runs even for
// a zero-trip loop, and runtime bounds are captured.

#ifndef HEADER
#define HEADER

#include <stdio.h>
#include <stdlib.h>

static void test_in_range_reverse_zero_trip(int n) {
  int i = 77, j = 88;
#pragma omp fuse
  {
    for (i = 0; i < 5; i++)
      ;
#pragma omp reverse
    for (j = 0; j < n; j++)
      ;
  }
  printf("in_range_zero: i=%d j=%d\n", i, j);
}

static void test_out_of_range_reverse_zero_trip(int n) {
  int i = 77, j = 88, k = 99;
#pragma omp fuse looprange(1, 2)
  {
    for (i = 0; i < 5; i++)
      ;
    for (j = 0; j < 6; j++)
      ;
#pragma omp reverse
    for (k = 0; k < n; k++)
      ;
  }
  printf("out_of_range_zero: i=%d j=%d k=%d\n", i, j, k);
}

static void test_out_of_range_reverse_captured_step(int start, int end,
                                                    int step) {
  int i = -1, j = -1, k = -1;
#pragma omp fuse looprange(1, 2)
  {
    for (i = 0; i < 4; i++)
      ;
    for (j = 0; j < 6; j++)
      ;
#pragma omp reverse
    for (k = start; k < end; k += step)
      ;
  }
  printf("out_of_range_step: i=%d j=%d k=%d\n", i, j, k);
}

static void test_in_range_reverse_captured_bound(int n) {
  int i = -1, j = -1;
#pragma omp fuse
  {
    for (i = 0; i < 10; i++)
      ;
#pragma omp reverse
    for (j = 0; j < n + 1; j++)
      ;
  }
  printf("captured_bound n=%d: i=%d j=%d\n", n, i, j);
}

int main(void) {
  test_in_range_reverse_zero_trip(0);
  test_out_of_range_reverse_zero_trip(0);
  test_out_of_range_reverse_captured_step(1, 10, 2);
  test_in_range_reverse_captured_bound(7);
  test_in_range_reverse_captured_bound(-1);
  return EXIT_SUCCESS;
}
#endif

// CHECK:      in_range_zero: i=5 j=0
// CHECK-NEXT: out_of_range_zero: i=5 j=6 k=0
// CHECK-NEXT: out_of_range_step: i=4 j=6 k=11
// CHECK-NEXT: captured_bound n=7: i=10 j=8
// CHECK-NEXT: captured_bound n=-1: i=10 j=0
