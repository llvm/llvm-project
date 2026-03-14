// RUN: %libomp-compile-and-run
// RUN: env KMP_LOCK_KIND=tas %libomp-run
// RUN: env KMP_LOCK_KIND=futex %libomp-run
#include <stdio.h>
#include "omp_testsuite.h"

omp_lock_t lck;

int test_omp_test_lock()
{
  int nr_threads_in_single = 0;
  int result = 0;
  int nr_iterations = 0;
  int i;

  omp_init_lock (&lck);
  #pragma omp parallel shared(lck)
  {
    #pragma omp for
    for (i = 0; i < LOOPCOUNT; i++) {
      while (!omp_test_lock (&lck))
      {};
      #pragma omp flush
      nr_threads_in_single++;
      #pragma omp flush
      nr_iterations++;
      nr_threads_in_single--;
      result = result + nr_threads_in_single;
      omp_unset_lock (&lck);
    }
  }
  omp_destroy_lock(&lck);
  return ((result == 0) && (nr_iterations == LOOPCOUNT));
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_test_lock()) {
      num_failed++;
    }
  }
  return num_failed;
}
