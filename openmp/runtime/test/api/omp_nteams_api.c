// RUN: %libomp-compile-and-run

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv)
{
  int iset, iget;
  iset = 5;
  omp_set_num_teams(iset);
  iget = omp_get_max_teams();
  if (iset != iget) {
    fprintf(stderr, "error: nteams-var set to %d, getter returned %d\n", iset, iget);
    exit(1);
  }
  iset = 7;
  omp_set_teams_thread_limit(iset);
  iget = omp_get_teams_thread_limit();
  if (iset != iget) {
    fprintf(stderr, "error: teams-thread-limit-var set to %d, getter returned %d\n", iset, iget);
    exit(1);
  }
  printf("passed\n");
  return 0;
}
