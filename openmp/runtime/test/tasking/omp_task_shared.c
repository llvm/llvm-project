// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

/* Utility function do spend some time in a loop */
int test_omp_task_imp_shared()
{
  int i;
  int k = 0;
  int result = 0;
  i=0;

  #pragma omp parallel
  {
    #pragma omp single
    for (k = 0; k < NUM_TASKS; k++) {
      #pragma omp task shared(i)
      {
        #pragma omp atomic
        i++;
        //this should be shared implicitly
      }
    }
  }
  result = i;
  return ((result == NUM_TASKS));
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_task_imp_shared()) {
      num_failed++;
    }
  }
  return num_failed;
}
