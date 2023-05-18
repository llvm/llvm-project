// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

/*
 * This test will hang if the nowait is not working properly
 *
 * It relies on a one thread skipping to the last single construct to
 * release the threads in the first three single constructs
 */
volatile int release;
volatile int count;

void wait_for_release_then_increment(int rank)
{
  fprintf(stderr, "Thread nr %d enters first section"
    " and waits.\n", rank);
  while (release == 0);
  #pragma omp atomic
  count++;
}

void release_and_increment(int rank)
{
  fprintf(stderr, "Thread nr %d sets release to 1\n", rank);
  release = 1;
  #pragma omp atomic
  count++;
}

int test_omp_single_nowait()
{
  release = 0;
  count = 0;

  #pragma omp parallel num_threads(4)
  {
    int rank;
    rank = omp_get_thread_num ();
    #pragma omp single nowait
    {
      wait_for_release_then_increment(rank);
    }
    #pragma omp single nowait
    {
      wait_for_release_then_increment(rank);
    }
    #pragma omp single nowait
    {
      wait_for_release_then_increment(rank);
    }

    #pragma omp single
    {
      release_and_increment(rank);
    }
  }
  // Check to make sure all four singles were executed
  return (count==4);
} /* end of check_single_nowait*/

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_single_nowait()) {
      num_failed++;
    }
  }
  return num_failed;
}
