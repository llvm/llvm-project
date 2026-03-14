// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

/*
 * This test will hang if the nowait is not working properly
 *
 * It relies on a thread skipping to the second sections construct to
 * release the threads in the first sections construct
 *
 * Also, since scheduling of sections is implementation defined, it is
 * necessary to have all four sections in the second sections construct
 * release the threads since we can't guarantee which section a single thread
 * will execute.
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
  #pragma omp flush(release)
  #pragma omp atomic
  count++;
}

int test_omp_sections_nowait()
{
  release = 0;
  count = 0;

  #pragma omp parallel num_threads(4)
  {
    int rank;
    rank = omp_get_thread_num ();
    #pragma omp sections nowait
    {
      #pragma omp section
      {
        wait_for_release_then_increment(rank);
      }
      #pragma omp section
      {
        wait_for_release_then_increment(rank);
      }
      #pragma omp section
      {
        wait_for_release_then_increment(rank);
      }
      #pragma omp section
      {
        fprintf(stderr, "Thread nr %d enters first sections and goes "
          "immediately to next sections construct to release.\n", rank);
        #pragma omp atomic
        count++;
      }
    }
    /* Begin of second sections environment */
    #pragma omp sections
    {
      #pragma omp section
      {
        release_and_increment(rank);
      }
      #pragma omp section
      {
        release_and_increment(rank);
      }
      #pragma omp section
      {
        release_and_increment(rank);
      }
      #pragma omp section
      {
        release_and_increment(rank);
      }
    }
  }
  // Check to make sure all eight sections were executed
  return (count==8);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_sections_nowait()) {
      num_failed++;
    }
  }
  return num_failed;
}
