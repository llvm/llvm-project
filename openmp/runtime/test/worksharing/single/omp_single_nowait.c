// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int my_iterations;
#pragma omp threadprivate(my_iterations)

int test_omp_single_nowait()
{
  int nr_iterations;
  int total_iterations = 0;
  int i;

  nr_iterations = 0;
  my_iterations = 0;

  #pragma omp parallel private(i)
  {
    for (i = 0; i < LOOPCOUNT; i++) {
      #pragma omp single nowait
      {
        #pragma omp atomic
        nr_iterations++;
      }
    }
  }

  #pragma omp parallel private(i)
  {
    my_iterations = 0;
    for (i = 0; i < LOOPCOUNT; i++) {
      #pragma omp single nowait
      {
        my_iterations++;
      }
    }
    #pragma omp critical
    {
      total_iterations += my_iterations;
    }

  }
  return ((nr_iterations == LOOPCOUNT) && (total_iterations == LOOPCOUNT));
} /* end of check_single_nowait*/

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_single_nowait()) {
      num_failed++;
    }
  }
  return num_failed;
}
