// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int test_omp_section_lastprivate()
{
  int i0 = -1;
  int sum = 0;
  int i;
  int sum0 = 0;
  int known_sum;

  i0 = -1;
  sum = 0;

  #pragma omp parallel
  {
    #pragma omp sections lastprivate(i0) private(i,sum0)
    {
      #pragma omp section
      {
        sum0 = 0;
        for (i = 1; i < 400; i++)
        {
          sum0 = sum0 + i;
          i0 = i;
        }
        #pragma omp critical
        {
          sum = sum + sum0;
        } /*end of critical*/
      } /* end of section */
      #pragma omp section
      {
        sum0 = 0;
        for(i = 400; i < 700; i++)
        {
          sum0 = sum0 + i;
          i0 = i;
        }
        #pragma omp critical
        {
          sum = sum + sum0;
        } /*end of critical*/
      }
      #pragma omp section
      {
        sum0 = 0;
        for(i = 700; i < 1000; i++)
        {
          sum0 = sum0 + i;
          i0 = i;
        }
        #pragma omp critical
        {
          sum = sum + sum0;
        } /*end of critical*/
      } /* end of section */
    } /* end of sections*/
  } /* end of parallel*/
  known_sum = (999 * 1000) / 2;
  return ((known_sum == sum) && (i0 == 999) );
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_section_lastprivate()) {
      num_failed++;
    }
  }
  return num_failed;
}
