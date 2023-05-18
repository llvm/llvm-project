// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_parallel_for_firstprivate()
{
  int sum ;
  int i2;
  int i;
  int known_sum;

  sum=0;
  i2=3;

  #pragma omp parallel for reduction(+:sum) private(i) firstprivate(i2)
  for (i = 1; i <= LOOPCOUNT; i++) {
    sum = sum + (i + i2);
  }

  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2 + i2 * LOOPCOUNT;
  return (known_sum == sum);
} /* end of check_parallel_for_fistprivate */

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_parallel_for_firstprivate()) {
      num_failed++;
    }
  }
  return num_failed;
}
