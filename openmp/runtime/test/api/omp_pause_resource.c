// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_pause_resource() {
  int fails, nthreads, my_dev;

  fails = 0;
  nthreads = 0;
  my_dev = omp_get_initial_device();

#pragma omp parallel
#pragma omp single
  nthreads = omp_get_num_threads();

  if (omp_pause_resource(omp_pause_soft, my_dev))
    fails++;

#pragma omp parallel shared(nthreads)
#pragma omp single
  nthreads = omp_get_num_threads();

  if (nthreads == 0)
    fails++;
  if (omp_pause_resource(omp_pause_hard, my_dev))
    fails++;
  nthreads = 0;

#pragma omp parallel shared(nthreads)
#pragma omp single
  nthreads = omp_get_num_threads();

  if (nthreads == 0)
    fails++;
  if (omp_pause_resource_all(omp_pause_soft))
    fails++;
  nthreads = 0;

#pragma omp parallel shared(nthreads)
#pragma omp single
  nthreads = omp_get_num_threads();

  if (nthreads == 0)
    fails++;
  return fails == 0;
}

int main() {
  int i;
  int num_failed = 0;

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_omp_pause_resource()) {
      num_failed++;
    }
  }
  return num_failed;
}
