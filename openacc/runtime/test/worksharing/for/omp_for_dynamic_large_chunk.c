// RUN: %libomp-compile
// RUN: env OMP_WAIT_POLICY=passive OMP_NUM_THREADS=32 %libomp-run 0 134217728 1 134217728
//
// This test makes sure that large chunks sizes are handled correctly
// including internal runtime calculations which incorporate the chunk size
// Only one thread should execute all iterations.
#include <stdio.h>
#include <stdlib.h>
#include "omp_testsuite.h"

typedef unsigned long long ull_t;

int main(int argc, char **argv) {
  int i, j, lb, ub, stride, nthreads, actual_nthreads, chunk;
  ull_t num_iters = 0;
  ull_t counted_iters = 0;
  int errs = 0;
  if (argc != 5) {
    fprintf(stderr, "error: incorrect number of arguments\n");
    fprintf(stderr, "usage: %s <lb> <ub> <stride> <chunk>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  lb = atoi(argv[1]);
  ub = atoi(argv[2]);
  stride = atoi(argv[3]);
  chunk = atoi(argv[4]);
  nthreads = omp_get_max_threads();
  if (lb >= ub) {
    fprintf(stderr, "error: lb must be less than ub\n");
    exit(EXIT_FAILURE);
  }
  if (stride <= 0) {
    fprintf(stderr, "error: stride must be positive integer\n");
    exit(EXIT_FAILURE);
  }
  if (chunk <= 0) {
    fprintf(stderr, "error: chunk must be positive integer\n");
    exit(EXIT_FAILURE);
  }
  for (i = lb; i < ub; i += stride)
    num_iters++;

  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp single
    actual_nthreads = omp_get_num_threads();

    if (actual_nthreads != nthreads) {
      printf("did not create enough threads, skipping test.\n");
    } else {
      #pragma omp for schedule(dynamic, chunk)
      for (i = lb; i < ub; i += stride) {
        counted_iters++;
      }
    }
  }

  // Check that the number of iterations executed is correct
  if (actual_nthreads == nthreads && counted_iters != num_iters) {
    fprintf(stderr, "error: wrong number of final iterations counted! "
                    "num_iters=%llu, counted_iters=%llu\n",
            num_iters, counted_iters);
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
