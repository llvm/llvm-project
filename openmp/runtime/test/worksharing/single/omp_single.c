// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_single()
{
  int nr_threads_in_single;
  int result;
  int nr_iterations;
  int i;

  nr_threads_in_single = 0;
  result = 0;
  nr_iterations = 0;

  #pragma omp parallel private(i)
  {
    for (i = 0; i < LOOPCOUNT; i++) {
      #pragma omp single
      {
        #pragma omp flush
        nr_threads_in_single++;
        #pragma omp flush
        nr_iterations++;
        nr_threads_in_single--;
        result = result + nr_threads_in_single;
      }
    }
  }
  return ((result == 0) && (nr_iterations == LOOPCOUNT));
} /* end of check_single*/

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_single()) {
      num_failed++;
    }
  }
  return num_failed;
}
