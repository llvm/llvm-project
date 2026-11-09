// RUN: %libomp-compile -fopenmp-version=60 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

// Loop variables after nested fuse constructs hold their loop-exit values
// (OpenMP 6.0): the inner fuse's finalization runs right after its fused loop,
// both when it follows and when it precedes the outer fused range.

#ifndef HEADER
#define HEADER

#include <stdio.h>
#include <stdlib.h>

static void test_nested_fuse_post(void) {
  int i = -1, j = -1, k = -1, l = -1, m = -1;
#pragma omp fuse looprange(1, 2)
  {
    for (i = 0; i < 4; i++)
      ;
    for (j = 0; j < 6; j++)
      ;
#pragma omp fuse looprange(1, 2)
    {
      for (k = 0; k < 8; k++)
        ;
      for (l = 0; l < 12; l++)
        ;
      for (m = 0; m < 5; m++)
        ;
    }
  }
  printf("post: i=%d j=%d k=%d l=%d m=%d\n", i, j, k, l, m);
}

static void test_nested_fuse_pre_read(void) {
  int i = -1, j = -1, k = -1, l = -1, m = -1;
  int count = 0;
#pragma omp fuse looprange(3, 2)
  {
#pragma omp fuse looprange(1, 2)
    {
      for (i = 0; i < 10; i++)
        ;
      for (j = 0; j < 15; j++)
        ;
      // Reads i: must observe its loop-exit value.
      for (k = 0; k < i; k++)
        count++;
    }
    for (l = 0; l < 4; l++)
      ;
    for (m = 0; m < 6; m++)
      ;
  }
  printf("pre: count=%d i=%d j=%d k=%d l=%d m=%d\n", count, i, j, k, l, m);
}

// split inside fuse is not supported yet; split follows the fuse here.
static void test_fuse_then_split(void) {
  int i = -1, j = -1, k = -1;
#pragma omp fuse looprange(1, 2)
  {
    for (i = 0; i < 4; i++)
      ;
    for (j = 0; j < 6; j++)
      ;
  }
#pragma omp split counts(3, omp_fill)
  for (k = 0; k < 10; k++)
    ;
  printf("split: i=%d j=%d k=%d\n", i, j, k);
}

int main(void) {
  test_nested_fuse_post();
  test_nested_fuse_pre_read();
  test_fuse_then_split();
  return EXIT_SUCCESS;
}
#endif

// CHECK:      post: i=4 j=6 k=8 l=12 m=5
// CHECK-NEXT: pre: count=10 i=10 j=15 k=10 l=4 m=6
// CHECK-NEXT: split: i=4 j=6 k=10
