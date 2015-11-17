// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <stdlib.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int test_omp_get_wtime()
{
  double start;
  double end;
  double measured_time;
  double wait_time = 5.0; 
  start = 0;
  end = 0;
  start = omp_get_wtime();
  my_sleep (wait_time); 
  end = omp_get_wtime();
  measured_time = end-start;
  return ((measured_time > 0.97 * wait_time) && (measured_time < 1.03 * wait_time)) ;
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_get_wtime()) {
      num_failed++;
    }
  }
  return num_failed;
}
