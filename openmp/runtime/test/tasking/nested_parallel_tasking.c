// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <omp.h>

/*
 * This test would hang when level instead of active level
 * used to push task state.
 */

int main()
{
  // If num_threads is changed to a value greater than 1, then the test passes
  #pragma omp parallel num_threads(1)
  {
    #pragma omp parallel
    printf("Hello World from thread %d\n", omp_get_thread_num());
  }

  printf("omp_num_threads: %d\n", omp_get_max_threads());

  #pragma omp parallel
  {
    #pragma omp master
    #pragma omp task default(none)
    {
      printf("%d is executing this task\n", omp_get_thread_num());
    }
  }

  printf("pass\n");
  return 0;
}
