// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_critical()
{
  int sum;
  int known_sum;

  sum=0;
  #pragma omp parallel
  {
    int mysum=0;
    int i;
    #pragma omp for
    for (i = 0; i < 1000; i++)
      mysum = mysum + i;

    #pragma omp critical
    sum = mysum +sum;
  }
  known_sum = 999 * 1000 / 2;
  return (known_sum == sum);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_critical()) {
      num_failed++;
    }
  }
  return num_failed;
}
