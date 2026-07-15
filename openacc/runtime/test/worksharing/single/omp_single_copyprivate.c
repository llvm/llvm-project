// RUN: %libomp-compile-and-run
#include "omp_testsuite.h"

#define DEBUG_TEST 0

int j;
#pragma omp threadprivate(j)

int test_omp_single_copyprivate()
{
  int result;
  int nr_iterations;

  result = 0;
  nr_iterations = 0;
  #pragma omp parallel num_threads(4)
  {
    int i;
    for (i = 0; i < LOOPCOUNT; i++)
    {
#if DEBUG_TEST
         int thread;
         thread = omp_get_thread_num ();
#endif
      #pragma omp single copyprivate(j)
      {
        nr_iterations++;
        j = i;
#if DEBUG_TEST
        printf ("thread %d assigns, j = %d, i = %d\n", thread, j, i);
#endif
      }
#if DEBUG_TEST
      #pragma omp barrier
#endif
      #pragma omp critical
      {
#if DEBUG_TEST
        printf ("thread = %d, j = %d, i = %d\n", thread, j, i);
#endif
        result = result + j - i;
      }
      #pragma omp barrier
    } /* end of for */
  } /* end of parallel */
  return ((result == 0) && (nr_iterations == LOOPCOUNT));
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_single_copyprivate()) {
      num_failed++;
    }
  }
  return num_failed;
}
