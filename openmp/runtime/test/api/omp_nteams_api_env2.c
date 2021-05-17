// RUN: %libomp-compile && env OMP_NUM_TEAMS=5 OMP_TEAMS_THREAD_LIMIT=7 %libomp-run

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv)
{
  int iset, iget;
  iset = 4; // should override OMP_NUM_TEAMS=5
  omp_set_num_teams(iset);
  iget = omp_get_max_teams();
  if (iset != iget) {
    fprintf(stderr, "error: nteams-var set to %d, getter returned %d\n", iset, iget);
    exit(1);
  }
  iset = 6; // should override OMP_TEAMS_THREAD_LIMIT=7
  omp_set_teams_thread_limit(iset);
  iget = omp_get_teams_thread_limit();
  if (iset != iget) {
    fprintf(stderr, "error: teams-thread-limit-var set to %d, getter returned %d\n", iset, iget);
    exit(1);
  }
  printf("passed\n");
  return 0;
}
