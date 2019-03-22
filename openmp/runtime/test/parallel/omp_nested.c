// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

/*
 * Test if the compiler supports nested parallelism
 * By Chunhua Liao, University of Houston
 * Oct. 2005
 */
int test_omp_nested()
{
#ifdef _OPENMP
  if (omp_get_max_threads() > 4)
    omp_set_num_threads(4);
  if (omp_get_max_threads() < 2)
    omp_set_num_threads(2);
#endif

  int counter = 0;
#ifdef _OPENMP
  omp_set_nested(1);
  omp_set_max_active_levels(omp_get_supported_active_levels());
#endif

  #pragma omp parallel shared(counter)
  {
    #pragma omp critical
    counter++;
    #pragma omp parallel
    {
      #pragma omp critical
      counter--;
    }
  }
  return (counter != 0);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_nested()) {
      num_failed++;
    }
  }
  return num_failed;
}
