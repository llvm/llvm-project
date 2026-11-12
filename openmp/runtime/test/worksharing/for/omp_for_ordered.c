// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

static int last_i = 0;

/* Utility function to check that i is increasing monotonically
   with each call */
static int check_i_islarger (int i)
{
  int islarger;
  islarger = (i > last_i);
  last_i = i;
  return (islarger);
}

int test_omp_for_ordered()
{
  int sum;
  int is_larger = 1;
  int known_sum;

  last_i = 0;
  sum = 0;

  #pragma omp parallel
  {
    int i;
    int my_islarger = 1;
    #pragma omp for schedule(static,1) ordered
    for (i = 1; i < 100; i++) {
      #pragma omp ordered
      {
        my_islarger = check_i_islarger(i) && my_islarger;
        sum = sum + i;
      }
    }
    #pragma omp critical
    {
      is_larger = is_larger && my_islarger;
    }
  }

  known_sum=(99 * 100) / 2;
  return ((known_sum == sum) && is_larger);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_for_ordered()) {
      num_failed++;
    }
  }
  return num_failed;
}
