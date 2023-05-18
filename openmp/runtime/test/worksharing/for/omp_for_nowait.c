// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"

/*
 * This test will hang if the nowait is not working properly.
 *
 * It relies on a thread skipping to the second for construct to
 * release the threads in the first for construct.
 *
 * Also, we use static scheduling to guarantee that one
 * thread will make it to the second for construct.
 */
volatile int release;
volatile int count;

void wait_for_release_then_increment(int rank)
{
  fprintf(stderr, "Thread nr %d enters first for construct"
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

int test_omp_for_nowait()
{
  release = 0;
  count = 0;

  #pragma omp parallel num_threads(4)
  {
    int rank;
    int i;

    rank = omp_get_thread_num();

    #pragma omp for schedule(static) nowait
    for (i = 0; i < 4; i++) {
      if (i < 3)
        wait_for_release_then_increment(rank);
      else {
        fprintf(stderr, "Thread nr %d enters first for and goes "
          "immediately to the next for construct to release.\n", rank);
        #pragma omp atomic
        count++;
      }
    }

    #pragma omp for schedule(static)
    for (i = 0; i < 4; i++) {
        release_and_increment(rank);
    }
  }
  return (count==8);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_for_nowait()) {
      num_failed++;
    }
  }
  return num_failed;
}
