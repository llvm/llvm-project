// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

/* Utility function to check that i is increasing monotonically
   with each call */
static int check_i_islarger (int i)
{
  static int last_i;
  int islarger;
  if (i==1)
    last_i=0;
  islarger = ((i >= last_i)&&(i - last_i<=1));
  last_i = i;
  return (islarger);
}

int test_omp_for_collapse()
{
  int is_larger = 1;

  #pragma omp parallel
  {
    int i,j;
    int my_islarger = 1;
    #pragma omp for private(i,j) schedule(static,1) collapse(2) ordered
    for (i = 1; i < 100; i++) {
      for (j =1; j <100; j++) {
        #pragma omp ordered
        my_islarger = check_i_islarger(i)&&my_islarger;
      }
    }
    #pragma omp critical
    is_larger = is_larger && my_islarger;
  }
  return (is_larger);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_for_collapse()) {
      num_failed++;
    }
  }
  return num_failed;
}
