// Test functionality of omp_set_num_teams() with setting
// environment variable as an upper bound. Test for negative
// value and value that is larger than the upper bound.

// RUN: %libomptarget-compile-generic -fopenmp-offload-mandatory
// RUN: env OMP_NUM_TEAMS_DEV_0=3 LIBOMPTARGET_INFO=16\
// RUN: %libomptarget-run-generic

// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO

#include <omp.h>
#include <stdio.h>

const int EXPECTED_NTEAMS = 3;

void omp_set_num_teams(int V);
int omp_get_max_teams(void);

int test_set_over_max(void) {
  int errors = 0;
  int n_devs;
  int curr_nteams = -1;

#pragma omp target map(tofrom : n_devs)
  { n_devs = omp_get_num_devices(); }

#pragma omp target device(0) map(tofrom : curr_nteams, errors)
  {
    omp_set_num_teams(3 + 1);
    curr_nteams = omp_get_max_teams();
    errors = errors + (curr_nteams != 3);

    omp_set_num_teams(-1);
    curr_nteams = omp_get_max_teams();
    errors = errors + (curr_nteams != 0);

    omp_set_num_teams(3);
    curr_nteams = omp_get_max_teams();
    errors = errors + (curr_nteams != 3);
  }
  return errors;
}

int main() {
  int errors = 0;
  errors = test_set_over_max();
  if (errors)
    printf("FAIL\n");
  else
    printf("PASS\n");
  return errors;
}

// CHECK: PASS