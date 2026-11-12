// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_master_3()
{
  int nthreads;
  int executing_thread;
  int tid_result = 0; /* counts up the number of wrong thread no. for
               the master thread. (Must be 0) */
  nthreads = 0;
  executing_thread = -1;

  #pragma omp parallel
  {
    #pragma omp master
    {
      int tid = omp_get_thread_num();
      if (tid != 0) {
        #pragma omp critical
        { tid_result++; }
      }
      #pragma omp critical
      {
        nthreads++;
      }
      executing_thread = omp_get_thread_num ();
    } /* end of master*/
  } /* end of parallel*/
  return ((nthreads == 1) && (executing_thread == 0) && (tid_result == 0));
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_master_3()) {
      num_failed++;
    }
  }
  return num_failed;
}
