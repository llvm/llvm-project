// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

static int last_i = 0;

int i;
#pragma omp threadprivate(i)

/* Variable ii is used to avoid problems with a threadprivate variable used as a loop
 * index. See test omp_threadprivate_for.
 */
static int ii;
#pragma omp threadprivate(ii)

/*!
  Utility function: returns true if the passed argument is larger than
  the argument of the last call of this function.
 */
static int check_i_islarger2(int i)
{
  int islarger;
  islarger = (i > last_i);
  last_i = i;
  return (islarger);
}

int test_omp_parallel_for_ordered()
{
  int sum;
  int is_larger;
  int known_sum;
  int i;

  sum = 0;
  is_larger = 1;
  last_i = 0;
  #pragma omp parallel for schedule(static,1) private(i) ordered
  for (i = 1; i < 100; i++) {
    ii = i;
    #pragma omp ordered
    {
      is_larger = check_i_islarger2 (ii) && is_larger;
      sum  = sum + ii;
    }
  }
  known_sum = (99 * 100) / 2;
  fprintf (stderr," known_sum = %d , sum = %d \n", known_sum, sum);
  fprintf (stderr," is_larger = %d\n", is_larger);
  return (known_sum == sum) && is_larger;
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_parallel_for_ordered()) {
      num_failed++;
    }
  }
  return num_failed;
}
