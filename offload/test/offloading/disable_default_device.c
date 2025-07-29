// RUN: %libomptarget-compile-generic
// RUN:   env OMP_TARGET_OFFLOAD=disabled %libomptarget-run-generic 2>&1 | %fcheck-generic

#include <omp.h>
#include <stdio.h>

// Sanity checks to make sure that this works and is thread safe.
int main() {
  // CHECK: number of devices 0
  printf("number of devices %d\n", omp_get_num_devices());
  // CHECK:initial device 0
  printf("initial device %d\n", omp_get_initial_device());
  // CHECK:default device 0
  printf("default device %d\n", omp_get_default_device());
  // CHECK: PASS
  if (omp_get_initial_device() == omp_get_default_device()) {
    printf("PASS\n");
    return 0;
  }
  return 1;
}
