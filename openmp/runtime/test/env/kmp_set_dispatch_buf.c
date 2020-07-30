// RUN: %libomp-compile
// RUN: env KMP_DISP_NUM_BUFFERS=0 %libomp-run
// RUN: env KMP_DISP_NUM_BUFFERS=1 %libomp-run
// RUN: env KMP_DISP_NUM_BUFFERS=3 %libomp-run
// RUN: env KMP_DISP_NUM_BUFFERS=4 %libomp-run
// RUN: env KMP_DISP_NUM_BUFFERS=7 %libomp-run
// RUN: %libomp-compile -DMY_SCHEDULE=guided
// RUN: env KMP_DISP_NUM_BUFFERS=1 %libomp-run
// RUN: env KMP_DISP_NUM_BUFFERS=3 %libomp-run
// RUN: env KMP_DISP_NUM_BUFFERS=4 %libomp-run
// RUN: env KMP_DISP_NUM_BUFFERS=7 %libomp-run
// UNSUPPORTED: clang-11
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <limits.h>
#include "omp_testsuite.h"

#define INCR 7
#define MY_MAX 200
#define MY_MIN -200
#define NUM_LOOPS 100
#ifndef MY_SCHEDULE
# define MY_SCHEDULE dynamic
#endif

int a, b, a_known_value, b_known_value;

int test_kmp_set_disp_num_buffers()
{
  int success = 1;
  a = 0;
  b = 0;
  // run many small dynamic loops to stress the dispatch buffer system
  #pragma omp parallel
  {
    int i,j;
    for (j = 0; j < NUM_LOOPS; j++) {
      #pragma omp for schedule(MY_SCHEDULE) nowait
      for (i = MY_MIN; i < MY_MAX; i+=INCR) {
        #pragma omp atomic
        a++;
      }
      #pragma omp for schedule(MY_SCHEDULE) nowait
      for (i = MY_MAX; i >= MY_MIN; i-=INCR) {
        #pragma omp atomic
        b++;
      }
    }
  }
  // detect failure
  if (a != a_known_value || b != b_known_value) {
    success = 0;
    printf("a = %d (should be %d), b = %d (should be %d)\n", a, a_known_value,
           b, b_known_value);
  }
  return success;
}

int main(int argc, char** argv)
{
  int i,j;
  int num_failed=0;

  // figure out the known values to compare with calculated result
  a_known_value = 0;
  b_known_value = 0;

  for (j = 0; j < NUM_LOOPS; j++) {
    for (i = MY_MIN; i < MY_MAX; i+=INCR)
      a_known_value++;
    for (i = MY_MAX; i >= MY_MIN; i-=INCR)
      b_known_value++;
  }

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_kmp_set_disp_num_buffers()) {
      num_failed++;
    }
  }
  return num_failed;
}
