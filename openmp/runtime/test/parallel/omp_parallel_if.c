// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_parallel_if()
{
  int i;
  int sum;
  int known_sum;
  int mysum;
  int control=1;

  sum =0;
  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2 ;
  #pragma omp parallel private(i) if(control==0)
  {
    mysum = 0;
    for (i = 1; i <= LOOPCOUNT; i++) {
      mysum = mysum + i;
    }
    #pragma omp critical
    {
      sum = sum + mysum;
    }
  }
  return (known_sum == sum);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_parallel_if()) {
      num_failed++;
    }
  }
  return num_failed;
}
