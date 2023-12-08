// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_get_num_threads()
{
  /* checks that omp_get_num_threads is equal to the number of
     threads */
  int nthreads_lib;
  int nthreads = 0;

  nthreads_lib = -1;

  #pragma omp parallel
  {
    #pragma omp critical
    {
      nthreads++;
    } /* end of critical */
    #pragma omp single
    {
      nthreads_lib = omp_get_num_threads ();
    }  /* end of single */
  } /* end of parallel */
  return (nthreads == nthreads_lib);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_get_num_threads()) {
      num_failed++;
    }
  }
  return num_failed;
}
