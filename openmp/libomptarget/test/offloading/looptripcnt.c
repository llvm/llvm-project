// RUN: %libomptarget-compile-generic && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 | %fcheck-generic -allow-empty -check-prefix=DEBUG
// REQUIRES: libomptarget-debug

/*
  Test for looptripcount being popped from runtime stack.
*/
#include <stdio.h>
#include <omp.h>
int main()
{
  int N = 128;
  int NN = 1024;
  int num_teams[NN];
  int num_threads[NN];

  printf("#pragma omp target teams distribute parallel for thread_limit(4)\n");
#pragma omp target teams distribute parallel for thread_limit(4)
  for (int j = 0; j< N; j++) {
    num_threads[j] = omp_get_num_threads();
    num_teams[j] = omp_get_num_teams();
  }
  printf("num_threads %d num_teams %d\n", num_threads[0], num_teams[0]);
// DEBUG: loop trip count is 128
  printf("#pragma omp target teams distribute parallel for\n");
#pragma omp target teams distribute parallel for
  for (int j = 0; j< N; j++) {
    num_threads[j] = omp_get_num_threads();
    num_teams[j] = omp_get_num_teams();
  }
  printf("num_threads %d num_teams %d\n", num_threads[0], num_teams[0]);
// DEBUG: loop trip count is 128
  return 0;
}
