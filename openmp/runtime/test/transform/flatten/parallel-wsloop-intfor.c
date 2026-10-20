// RUN: %libomp-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

// Flatten changes which iterations a worksharing loop assigns to which
// thread. With schedule(static,1) and two threads:
//   * without flatten, the outer loop (2 iterations) is distributed
//   * with flatten, the product space (4 iterations) is distributed
// Results are printed per thread after the parallel region so FileCheck
// is not racy.

#ifndef HEADER
#define HEADER

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

enum { NThreads = 2, MaxIters = 4 };

static void dump(const char *tag, int count[NThreads],
                 int pairs[NThreads][MaxIters][2]) {
  printf("%s\n", tag);
  for (int t = 0; t < NThreads; ++t) {
    printf("tid=%d count=%d\n", t, count[t]);
    for (int k = 0; k < count[t]; ++k)
      printf("tid=%d i=%d j=%d\n", t, pairs[t][k][0], pairs[t][k][1]);
  }
}

int main() {
  int count[NThreads];
  int pairs[NThreads][MaxIters][2];

  count[0] = count[1] = 0;
#pragma omp parallel for schedule(static, 1) num_threads(2)
#pragma omp flatten
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      int t = omp_get_thread_num();
      int c = count[t]++;
      pairs[t][c][0] = i;
      pairs[t][c][1] = j;
    }
  dump("with-flatten", count, pairs);

  count[0] = count[1] = 0;
#pragma omp parallel for schedule(static, 1) num_threads(2)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      int t = omp_get_thread_num();
      int c = count[t]++;
      pairs[t][c][0] = i;
      pairs[t][c][1] = j;
    }
  dump("without-flatten", count, pairs);

  return EXIT_SUCCESS;
}

#endif /* HEADER */

// Flattened product space has 4 iterations; static,1 with 2 threads gives
// each thread 2 chunks: (0,0)+(1,0) and (0,1)+(1,1).
// CHECK:      with-flatten
// CHECK-NEXT: tid=0 count=2
// CHECK-NEXT: tid=0 i=0 j=0
// CHECK-NEXT: tid=0 i=1 j=0
// CHECK-NEXT: tid=1 count=2
// CHECK-NEXT: tid=1 i=0 j=1
// CHECK-NEXT: tid=1 i=1 j=1

// Without flatten the outer loop has 2 iterations, so each thread gets one
// value of i and both j.
// CHECK:      without-flatten
// CHECK-NEXT: tid=0 count=2
// CHECK-NEXT: tid=0 i=0 j=0
// CHECK-NEXT: tid=0 i=0 j=1
// CHECK-NEXT: tid=1 count=2
// CHECK-NEXT: tid=1 i=1 j=0
// CHECK-NEXT: tid=1 i=1 j=1
