// RUN: %libomp-compile-and-run
/*
 * Threadprivate is tested in 2 ways:
 * 1. The global variable declared as threadprivate should have
 *  local copy for each thread. Otherwise race condition and
 *  wrong result.
 * 2. If the value of local copy is retained for the two adjacent
 *  parallel regions
 */
#include "omp_testsuite.h"
#include <stdlib.h>
#include <stdio.h>

static int sum0=0;
static int myvalue = 0;

#pragma omp threadprivate(sum0)
#pragma omp threadprivate(myvalue)

int test_omp_threadprivate()
{
  int sum = 0;
  int known_sum;
  int i;
  int iter;
  int *data;
  int size;
  int num_failed = 0;
  int my_random;
  omp_set_dynamic(0);

  #pragma omp parallel private(i)
  {
    sum0 = 0;
    #pragma omp for
    for (i = 1; i <= LOOPCOUNT; i++) {
      sum0 = sum0 + i;
    } /*end of for*/
    #pragma omp critical
    {
      sum = sum + sum0;
    } /*end of critical */
  } /* end of parallel */
  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
  if (known_sum != sum ) {
    fprintf (stderr, " known_sum = %d, sum = %d\n", known_sum, sum);
  }

  /* the next parallel region is just used to get the number of threads*/
  omp_set_dynamic(0);
  #pragma omp parallel
  {
    #pragma omp master
    {
      size=omp_get_num_threads();
      data=(int*) malloc(size*sizeof(int));
    }
  }/* end parallel*/

  srand(45);
  for (iter = 0; iter < 100; iter++) {
    my_random = rand(); /* random number generator is
                 called inside serial region*/

    /* the first parallel region is used to initialize myvalue
       and the array with my_random+rank */
    #pragma omp parallel
    {
      int rank;
      rank = omp_get_thread_num ();
      myvalue = data[rank] = my_random + rank;
    }

    /* the second parallel region verifies that the
       value of "myvalue" is retained */
    #pragma omp parallel reduction(+:num_failed)
    {
      int rank;
      rank = omp_get_thread_num ();
      num_failed = num_failed + (myvalue != data[rank]);
      if(myvalue != data[rank]) {
        fprintf (stderr, " myvalue = %d, data[rank]= %d\n",
          myvalue, data[rank]);
      }
    }
  }
  free (data);
  return (known_sum == sum) && !num_failed;
} /* end of check_threadprivate*/

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_threadprivate()) {
      num_failed++;
    }
  }
  return num_failed;
}
