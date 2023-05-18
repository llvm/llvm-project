// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

static omp_nest_lock_t lck;

int test_omp_test_nest_lock()
{
  int nr_threads_in_single = 0;
  int result = 0;
  int nr_iterations = 0;
  int i;

  omp_init_nest_lock (&lck);
  #pragma omp parallel shared(lck)
  {
    #pragma omp for
    for (i = 0; i < LOOPCOUNT; i++)
    {
      /*omp_set_lock(&lck);*/
      while(!omp_test_nest_lock (&lck))
      {};
      #pragma omp flush
      nr_threads_in_single++;
      #pragma omp flush
      nr_iterations++;
      nr_threads_in_single--;
      result = result + nr_threads_in_single;
      omp_unset_nest_lock (&lck);
    }
  }
  omp_destroy_nest_lock (&lck);
  return ((result == 0) && (nr_iterations == LOOPCOUNT));
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_test_nest_lock()) {
      num_failed++;
    }
  }
  return num_failed;
}
