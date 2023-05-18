// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_master()
{
  int nthreads;
  int executing_thread;

  nthreads = 0;
  executing_thread = -1;

  #pragma omp parallel
  {
    #pragma omp master
    {
      #pragma omp critical
      {
        nthreads++;
      }
      executing_thread = omp_get_thread_num();
    } /* end of master*/
  } /* end of parallel*/
  return ((nthreads == 1) && (executing_thread == 0));
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_master()) {
      num_failed++;
    }
  }
  return num_failed;
}
