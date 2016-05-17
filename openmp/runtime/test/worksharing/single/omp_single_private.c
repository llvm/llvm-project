// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

int myit = 0;
#pragma omp threadprivate(myit)
int myresult = 0;
#pragma omp threadprivate(myresult)

int test_omp_single_private()
{
  int nr_threads_in_single;
  int result;
  int nr_iterations;
  int i;

  myit = 0;
  nr_threads_in_single = 0;
  nr_iterations = 0;
  result = 0;

  #pragma omp parallel private(i)
  {
    myresult = 0;
    myit = 0;
    for (i = 0; i < LOOPCOUNT; i++) {
      #pragma omp single private(nr_threads_in_single) nowait
      {
        nr_threads_in_single = 0;
        #pragma omp flush
        nr_threads_in_single++;
        #pragma omp flush
        myit++;
        myresult = myresult + nr_threads_in_single;
      }
    }
    #pragma omp critical
    {
      result += nr_threads_in_single;
      nr_iterations += myit;
    }
  }
  return ((result == 0) && (nr_iterations == LOOPCOUNT));
} /* end of check_single private */

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_single_private()) {
      num_failed++;
    }
  }
  return num_failed;
}
