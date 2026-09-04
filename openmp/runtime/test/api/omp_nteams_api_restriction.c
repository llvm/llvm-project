// RUN: %libomp-compile-and-run 2>&1 | FileCheck %s
// Verify that omp_set_num_teams() and omp_set_teams_thread_limit() are
// ignored when called from within a parallel or teams region.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv) {
  int err = 0;

  // Set initial values outside any region.
  omp_set_num_teams(5);
  omp_set_teams_thread_limit(7);

  if (omp_get_max_teams() != 5) {
    fprintf(stderr, "error: nteams-var not set correctly\n");
    exit(1);
  }
  if (omp_get_teams_thread_limit() != 7) {
    fprintf(stderr, "error: teams-thread-limit-var not set correctly\n");
    exit(1);
  }

  // Call from inside a teams region -- should be ignored.
  // Use num_teams(1) to ensure exactly one team and deterministic warning
  // count.
#pragma omp teams num_teams(1)
  {
    omp_set_num_teams(99);
    omp_set_teams_thread_limit(99);
  }

  if (omp_get_max_teams() != 5) {
    fprintf(stderr, "error: nteams-var modified inside teams region\n");
    err++;
  }
  if (omp_get_teams_thread_limit() != 7) {
    fprintf(stderr, "error: teams-thread-limit-var modified inside teams "
                    "region\n");
    err++;
  }

  // Call from inside a parallel region -- should be ignored.
#pragma omp parallel num_threads(1)
  {
    omp_set_num_teams(99);
    omp_set_teams_thread_limit(99);
  }

  if (omp_get_max_teams() != 5) {
    fprintf(stderr, "error: nteams-var modified inside parallel region\n");
    err++;
  }
  if (omp_get_teams_thread_limit() != 7) {
    fprintf(stderr, "error: teams-thread-limit-var modified inside parallel "
                    "region\n");
    err++;
  }

  if (err == 0) {
    printf("passed\n");
  }
  return err;
}

// CHECK: OMP: Warning{{.*}}omp_set_num_teams{{.*}}call ignored
// CHECK: OMP: Warning{{.*}}omp_set_teams_thread_limit{{.*}}call ignored
// CHECK: OMP: Warning{{.*}}omp_set_num_teams{{.*}}call ignored
// CHECK: OMP: Warning{{.*}}omp_set_teams_thread_limit{{.*}}call ignored
// CHECK: passed
