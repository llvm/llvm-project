// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

/*
 * Checks that false is returned when called from serial region
 * and true is returned when called within parallel region.
 */
int test_omp_in_parallel()
{
  int serial;
  int isparallel;

  serial = 1;
  isparallel = 0;
  serial = omp_in_parallel();

  #pragma omp parallel
  {
    #pragma omp single
    {
      isparallel = omp_in_parallel();
    }
  }
  return (!(serial) && isparallel);
}

int main()
{
  int i;
  int num_failed=0;

  // the test requires more than 1 thread to pass
  omp_set_dynamic(0); // disable dynamic adjustment of threads
  if (omp_get_max_threads() == 1)
    omp_set_num_threads(2); // set 2 threads if no HW resources available

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_in_parallel()) {
      num_failed++;
    }
  }
  return num_failed;
}
