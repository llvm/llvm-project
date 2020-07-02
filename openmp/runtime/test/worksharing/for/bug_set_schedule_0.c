// RUN: %libomp-compile-and-run

// gcc 9/10 use GOMP_parallel_loop_maybe_nonmonotonic_runtime, not implemented
// Should be removed once the functions are implemented
// XFAIL: gcc-9, gcc-10

#include <stdio.h>
#include <omp.h>
#include "omp_testsuite.h"

/* Test that the chunk size is set to default (1) when
   chunk size <= 0 is specified */
int a = 0;

int test_set_schedule_0()
{
  int i;
  a = 0;
  omp_set_schedule(omp_sched_dynamic,0);

  #pragma omp parallel
  {
    #pragma omp for schedule(runtime)
    for(i = 0; i < 10; i++) {
      #pragma omp atomic
      a++;
      if(a > 10)
        exit(1);
    }
  }
  return a==10;
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_set_schedule_0()) {
      num_failed++;
    }
  }
  return num_failed;
}
