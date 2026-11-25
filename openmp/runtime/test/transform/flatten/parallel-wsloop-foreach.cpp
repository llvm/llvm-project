// RUN: %libomp-cxx-compile -fopenmp-version=61 && %libomp-run \
// RUN:   | FileCheck %s --match-full-lines

#ifndef HEADER
#define HEADER

#include <cstdio>
#include <cstdlib>
#include <omp.h>

enum { NThreads = 2, MaxIters = 4 };

static void dump(const char *Tag, int Count[NThreads],
                 int Pairs[NThreads][MaxIters][2]) {
  printf("%s\n", Tag);
  for (int T = 0; T < NThreads; ++T) {
    printf("tid=%d count=%d\n", T, Count[T]);
    for (int K = 0; K < Count[T]; ++K)
      printf("tid=%d i=%d j=%d\n", T, Pairs[T][K][0], Pairs[T][K][1]);
  }
}

int main() {
  int Is[] = {0, 1};
  int Js[] = {0, 1};
  int Count[NThreads] = {};
  int Pairs[NThreads][MaxIters][2];

#pragma omp parallel for schedule(static, 1) num_threads(2)
#pragma omp flatten
  for (int i : Is)
    for (int j : Js) {
      int T = omp_get_thread_num();
      int C = Count[T]++;
      Pairs[T][C][0] = i;
      Pairs[T][C][1] = j;
    }
  dump("with-flatten", Count, Pairs);

  return EXIT_SUCCESS;
}

#endif /* HEADER */

// The flattened range-based product has four iterations. Static scheduling
// gives each thread alternating logical iterations.
// CHECK:      with-flatten
// CHECK-NEXT: tid=0 count=2
// CHECK-NEXT: tid=0 i=0 j=0
// CHECK-NEXT: tid=0 i=1 j=0
// CHECK-NEXT: tid=1 count=2
// CHECK-NEXT: tid=1 i=0 j=1
// CHECK-NEXT: tid=1 i=1 j=1
