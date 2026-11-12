// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_parallel_sections_firstprivate()
{
  int sum;
  int sum0;
  int known_sum;

  sum =7;
  sum0=11;

  #pragma omp parallel sections firstprivate(sum0)
  {
    #pragma omp section
    {
      #pragma omp critical
      {
        sum= sum+sum0;
      }
    }
    #pragma omp section
    {
      #pragma omp critical
      {
        sum= sum+sum0;
      }
    }
    #pragma omp section
    {
      #pragma omp critical
      {
        sum= sum+sum0;
      }
    }
  }

  known_sum=11*3+7;
  return (known_sum==sum);
} /* end of check_section_firstprivate*/

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_parallel_sections_firstprivate()) {
      num_failed++;
    }
  }
  return num_failed;
}
