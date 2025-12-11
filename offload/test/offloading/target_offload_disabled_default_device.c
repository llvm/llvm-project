// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=disabled %libomptarget-run-generic 2>&1 |
// %fcheck-generic
//
// Test that setting default device before disabling offload doesn't crash

#include <omp.h>
#include <stdio.h>

int main() {
  // Set high default device number
  omp_set_default_device(5);

  // This simulates OMP_TARGET_OFFLOAD=disabled being set after device is chosen
  // In practice, the environment variable is read at runtime init

  // CHECK: num_devices: 0
  printf("num_devices: %d\n", omp_get_num_devices());

  // CHECK: initial_device: 0
  printf("initial_device: %d\n", omp_get_initial_device());

  // CHECK: default_device: 0
  printf("default_device: %d\n", omp_get_default_device());

  // Target region should execute on host
  int result = -1;
#pragma omp target map(from : result)
  {
    result = omp_get_device_num();
  }

  // CHECK: executed_on: 0
  printf("executed_on: %d\n", result);

  // CHECK: PASS
  if (result == omp_get_initial_device() &&
      omp_get_default_device() == omp_get_initial_device()) {
    printf("PASS\n");
    return 0;
  }

  printf("FAIL\n");
  return 1;
}
