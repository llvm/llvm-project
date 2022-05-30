// Check that omp atomic compare handles signedness of integer comparisons
// correctly.
//
// At one time, a bug sometimes reversed the signedness.

// RUN: %libomp-compile -fopenmp-version=51
// RUN: %libomp-run | FileCheck %s

// High parallelism increases our chances of detecting a lack of atomicity.
#define NUM_THREADS_TRY 256

#include <limits.h>
#include <omp.h>
#include <stdio.h>

int main() {
  //      CHECK: signed: num_threads=[[#NUM_THREADS:]]{{$}}
  // CHECK-NEXT: signed: xs=[[#NUM_THREADS-1]]{{$}}
  int xs = -1;
  int numThreads;
  #pragma omp parallel for num_threads(NUM_THREADS_TRY)
  for (int i = 0; i < omp_get_num_threads(); ++i) {
    #pragma omp atomic compare
    if (xs < i) { xs = i; }
    if (i == 0)
      numThreads = omp_get_num_threads();
  }
  printf("signed: num_threads=%d\n", numThreads);
  printf("signed: xs=%d\n", xs);

  // CHECK-NEXT: unsigned: xu=0x0{{$}}
  unsigned xu = UINT_MAX;
  #pragma omp parallel for num_threads(NUM_THREADS_TRY)
  for (int i = 0; i < omp_get_num_threads(); ++i) {
    #pragma omp atomic compare
    if (xu > i) { xu = i; }
  }
  printf("unsigned: xu=0x%x\n", xu);
  return 0;
}
