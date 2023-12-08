// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <omp.h>
#include "omp_my_sleep.h"

/*
 * This test creates tasks that themselves create a new task.
 * The runtime has to take care that they are correctly freed.
 */

int main()
{
  #pragma omp task
  {
    #pragma omp task
    {
      my_sleep( 0.1 );
    }
  }

  #pragma omp parallel num_threads(2)
  {
    #pragma omp single
    #pragma omp task
    {
      #pragma omp task
      {
        my_sleep( 0.1 );
      }
    }
  }

  printf("pass\n");
  return 0;
}
