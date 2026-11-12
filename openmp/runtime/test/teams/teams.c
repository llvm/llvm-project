// RUN: %libomp-compile-and-run
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8
// UNSUPPORTED: icc, clang

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_TEAMS 2
#define NUM_THREADS_PER_TEAM 3

int main(int argc, char** argv) {
  #pragma omp teams num_teams(NUM_TEAMS)
  {
    int i;
    int members[NUM_THREADS_PER_TEAM];
    // Only an upper bound is guaranteed for number of teams
    int nteams = omp_get_num_teams();
    if (nteams > NUM_TEAMS) {
      fprintf(stderr, "error: too many teams: %d\n", nteams);
      exit(1);
    }
    for (i = 0; i < NUM_THREADS_PER_TEAM; ++i)
      members[i] = -1;
    #pragma omp parallel num_threads(NUM_THREADS_PER_TEAM) private(i)
    {
      int tid = omp_get_thread_num();
      int team_id = omp_get_team_num();
      int nthreads = omp_get_num_threads();
      if (nthreads != NUM_THREADS_PER_TEAM) {
        fprintf(stderr, "error: detected number of threads (%d) is not %d\n",
                nthreads, NUM_THREADS_PER_TEAM);
        exit(1);
      }
      if (tid < 0 || tid >= nthreads) {
        fprintf(stderr, "error: thread id is out of range: %d\n", tid);
        exit(1);
      }
      if (team_id < 0 || team_id > omp_get_num_teams()) {
        fprintf(stderr, "error: team id is out of range: %d\n", team_id);
        exit(1);
      }
      members[omp_get_thread_num()] = 1;
      #pragma omp barrier
      #pragma omp single
      {
        for (i = 0; i < NUM_THREADS_PER_TEAM; ++i) {
          if (members[i] != 1) {
            fprintf(stderr, "error: worker %d not flagged\n", i);
            exit(1);
          }
        }
      }
    }
  }
  return 0;
}
