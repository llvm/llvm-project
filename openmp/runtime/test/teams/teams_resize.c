// RUN: %libomp-compile && env OMP_DYNAMIC=true KMP_DYNAMIC_MODE=random %libomp-run
// UNSUPPORTED: icc

// This is a super simple unit test to see that teams behave properly when
// parallel regions inside the teams construct cannot allocate teams of
// thread_limit size.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_TIMES 10

int main(int argc, char **argv) {
  int num_procs = omp_get_max_threads();
  int num_teams, thread_limit, i;
  num_teams = 2;
  thread_limit = num_procs / num_teams;
  for (i = 0; i < NUM_TIMES; ++i) {
#pragma omp target teams num_teams(num_teams) thread_limit(thread_limit)
    {
#pragma omp parallel num_threads(thread_limit)
      {
        int my_num_threads = omp_get_num_threads();
        int my_num_teams = omp_get_num_teams();
        int my_team_id = omp_get_team_num();
        int my_thread_id = omp_get_thread_num();
        if (my_num_teams < 0 || my_num_teams > num_teams) {
          fprintf(stderr, "error: my_num_teams (%d) invalid\n", my_num_teams);
          exit(1);
        }
        if (my_team_id < 0 || my_team_id >= my_num_teams) {
          fprintf(stderr, "error: my_team_id (%d) invalid (nteams = %d)\n",
                  my_team_id, my_num_teams);
          exit(1);
        }
        if (my_thread_id < 0 || my_thread_id >= my_num_threads) {
          fprintf(stderr,
                  "error: my_thread_id (%d) invalid (my_num_threads = %d)\n",
                  my_thread_id, my_num_threads);
          exit(1);
        }
      }
    }
  }
  return 0;
}
