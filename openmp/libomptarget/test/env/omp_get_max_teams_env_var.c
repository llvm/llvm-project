// Test functionality of omp_get_max_teams() with setting
// environment variable to 2 GPU devices. If there's only
// one GPU device, remove the device 1 if statement.

// RUN: %libomptarget-compile-generic -fopenmp-offload-mandatory
// RUN: env OMP_NUM_TEAMS_DEV_0=5 OMP_NUM_TEAMS_DEV_1=-1 \
// RUN: %libomptarget-run-generic

// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO

#include <omp.h>
#include <stdio.h>

const int EXPECTED_NTEAMS_DEV_0 = 5;
const int EXPECTED_NTEAMS_DEV_1 = 0;

int omp_get_max_teams(void);

int test_nteams_var_env(void) {
  int errors = 0;
  int device_id;
  int n_devs;
  int curr_nteams = -1;
#pragma omp target map(tofrom : n_devs)
  { n_devs = omp_get_num_devices(); }

  for (int i = 0; i < n_devs; i++) {
#pragma omp target device(i) map(tofrom : curr_nteams, device_id, errors)
    {
      device_id = omp_get_device_num();
      errors = errors + (device_id != i);
      curr_nteams = omp_get_max_teams();
      if (device_id == 0) {
        errors = errors + (curr_nteams != EXPECTED_NTEAMS_DEV_0);
      } // device 0
      if (device_id == 1) {
        errors = errors + (curr_nteams != EXPECTED_NTEAMS_DEV_1);
      } // device 1
    }
    printf("device: %d nteams: %d\n", device_id, curr_nteams);
  }
  return errors;
}

int main() {
  int errors = 0;
  errors = test_nteams_var_env();
  if (errors)
    printf("FAIL\n");
  else
    printf("PASS\n");
  return errors;
}

// CHECK: PASS