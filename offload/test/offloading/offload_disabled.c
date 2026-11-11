// Check that no liboffload function is invoked without liboffload being
// initialized when offloading is disabled.

// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=disabled %libomptarget-run-generic 2>&1 | \
// RUN:     %fcheck-generic --implicit-check-not="omptarget error"

#include <omp.h>
#include <stdio.h>

int main() {
  int IsHost = 0;

#pragma omp target map(from : IsHost)
  IsHost = omp_is_initial_device();

  // CHECK: number of devices 0
  printf("number of devices %d\n", omp_get_num_devices());
  // CHECK: PASS
  if (IsHost) {
    printf("PASS\n");
    return 0;
  }
  return 1;
}
