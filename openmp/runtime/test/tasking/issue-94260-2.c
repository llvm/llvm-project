// RUN: %libomp-compile-and-run

#include <stdio.h>
#include <omp.h>

int test_omp_parallel_num_threads() {
  int num_failed;
  int threads;
  int nthreads;
  int max_threads = 0;

  num_failed = 0;
#pragma omp task
  {}

/* first we check how many threads are available */
#pragma omp parallel
  {
#pragma omp task
    {}
#pragma omp master
    max_threads = omp_get_num_threads();
  }

  /* we increase the number of threads from one to maximum:*/
  for (threads = 1; threads <= max_threads; threads++) {
    nthreads = 0;
#pragma omp parallel reduction(+ : num_failed) num_threads(threads)
    {
#pragma omp task
      {}
      num_failed = num_failed + !(threads == omp_get_num_threads());
#pragma omp atomic
      nthreads += 1;
    }
    num_failed = num_failed + !(nthreads == threads);
  }
  return (!num_failed);
}

int main() {
  int i;
  int num_failed = 0;

  for (i = 0; i < 100; i++) {
    if (!test_omp_parallel_num_threads()) {
      num_failed++;
    }
  }
  return num_failed;
}
