// RUN: %libomp-compile -DMY_SCHEDULE=static && %libomp-run
// RUN: %libomp-compile -DMY_SCHEDULE=dynamic && %libomp-run
// RUN: %libomp-compile -DMY_SCHEDULE=guided && %libomp-run

// Only works with Intel Compiler since at least version 15.0
// XFAIL: gcc, clang

/*
 * Test that large bounds are handled properly and calculations of
 * loop iterations don't accidently overflow
 */
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <limits.h>
#include "omp_testsuite.h"

#define INCR 50000000
#define MY_MAX 2000000000
#define MY_MIN -2000000000
#ifndef MY_SCHEDULE
# define MY_SCHEDULE static
#endif

int a, b, a_known_value, b_known_value;

int test_omp_for_bigbounds()
{
  a = 0;
  b = 0;
  #pragma omp parallel
  {
    int i;
    #pragma omp for schedule(MY_SCHEDULE)
    for (i = INT_MIN; i < MY_MAX; i+=INCR) {
        #pragma omp atomic
        a++;
    }
    #pragma omp for schedule(MY_SCHEDULE)
    for (i = INT_MAX; i >= MY_MIN; i-=INCR) {
        #pragma omp atomic
        b++;
    }
  }
  printf("a = %d (should be %d), b = %d (should be %d)\n", a, a_known_value, b, b_known_value);
  return (a == a_known_value && b == b_known_value);
}

int main()
{
  int i;
  int num_failed=0;

  a_known_value = 0;
  for (i = INT_MIN; i < MY_MAX; i+=INCR) {
      a_known_value++;
  }

  b_known_value = 0;
  for (i = INT_MAX; i >= MY_MIN; i-=INCR) {
      b_known_value++;
  }

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_for_bigbounds()) {
      num_failed++;
    }
  }
  return num_failed;
}
