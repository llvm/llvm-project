// RUN: %libomp-compile
// RUN: env OMP_NUM_THREADS=4 %libomp-run 5 5005 500 1000000000
// It fails using gcc compilers because the gcc compiler does not use any
// runtime interface to calculate the iterations for static loop schedule
// Hence, the runtime is never involved.
// XFAIL: gcc
//
// This test makes sure that large chunks sizes are handled correctly
// including internal runtime calculations which incorporate the chunk size
#include <stdio.h>
#include <stdlib.h>
#include "omp_testsuite.h"

#ifndef DEBUG_OUTPUT
#define DEBUG_OUTPUT 0
#endif

// Used in qsort() to compare integers
int compare_ints(const void *v1, const void *v2) {
  int i1 = *(const int *)v1;
  int i2 = *(const int *)v2;
  return i1 - i2;
}

int main(int argc, char **argv) {
  int i, j, lb, ub, stride, nthreads, chunk;
  int num_iters = 0;
  int counted_iters = 0;
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
  // Thread private record of iterations each thread performed
  int *iters = (int *)malloc(sizeof(int) * nthreads * num_iters);
  // This will be the list of all iteration performed by every thread
  int *final_iters = (int *)malloc(sizeof(int) * nthreads * num_iters);
  for (i = 0; i < nthreads * num_iters; ++i) {
    iters[i] = -1;
    final_iters[i] = -1;
  }

  #pragma omp parallel num_threads(nthreads)
  {
    int j = 0;
    int *my_iters = iters + omp_get_thread_num() * num_iters;
    #pragma omp for schedule(static, chunk)
    for (i = lb; i < ub; i += stride) {
      #pragma omp atomic
      counted_iters++;
      my_iters[j++] = i;
    }
  }

  // Put all iterations into final_iters then sort it from lowest to highest
  for (i = 0, j = 0; i < nthreads * num_iters; ++i) {
    if (iters[i] != -1)
      final_iters[j++] = iters[i];
  }
  if (j != counted_iters) {
    fprintf(stderr, "error: wrong number of final iterations counted!\n");
    exit(EXIT_FAILURE);
  }
  qsort(final_iters, j, sizeof(int), compare_ints);

  // Check for the right number of iterations
  if (counted_iters != num_iters) {
    fprintf(stderr, "error: wrong number of iterations executed. Expected %d "
                    "but executed %d\n",
            num_iters, counted_iters);
    exit(EXIT_FAILURE);
  }

#if DEBUG_OUTPUT
  for (i = 0; i < num_iters; ++i)
    printf("final_iters[%d] = %d\n", i, final_iters[i]);
#endif

  // Check that the iterations performed were correct
  for (i = lb, j = 0; i < ub; i += stride, ++j) {
    if (final_iters[j] != i) {
      fprintf(stderr,
              "error: iteration j=%d i=%d is incorrect. Expect %d but see %d\n",
              j, i, i, final_iters[j]);
      exit(EXIT_FAILURE);
    }
  }

  free(iters);
  free(final_iters);
  return EXIT_SUCCESS;
}
