// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_parallel_default()
{
  int i;
  int sum;
  int mysum;
  int known_sum;
  sum =0;
  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2 ;

  #pragma omp parallel default(shared) private(i) private(mysum)
  {
    mysum = 0;
    #pragma omp for
    for (i = 1; i <= LOOPCOUNT; i++) {
      mysum = mysum + i;
    }
    #pragma omp critical
    {
      sum = sum + mysum;
    }   /* end of critical */
  }   /* end of parallel */
  if (known_sum != sum) {
    fprintf(stderr, "KNOWN_SUM = %d; SUM = %d\n", known_sum, sum);
  }
  return (known_sum == sum);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_parallel_default()) {
      num_failed++;
    }
  }
  return num_failed;
}
