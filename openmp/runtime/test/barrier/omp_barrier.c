// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int test_omp_barrier()
{
  int result1;
  int result2;
  result1 = 0;
  result2 = 0;

  #pragma omp parallel
  {
    int rank;
    rank = omp_get_thread_num ();
    if (rank ==1) {
      my_sleep(((double)SLEEPTIME)/REPETITIONS); // give 1 sec to whole test
      result2 = 3;
    }
    #pragma omp barrier
    if (rank == 2) {
      result1 = result2;
    }
  }
  return (result1 == 3);
}

int main()
{
  int i;
  int num_failed=0;

#ifdef _OPENMP
  omp_set_dynamic(0); // prevent runtime to change number of threads
  omp_set_num_threads(4); // the test expects at least 3 threads
  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_barrier()) {
      num_failed++;
    }
  }
#endif
  return num_failed;
}
