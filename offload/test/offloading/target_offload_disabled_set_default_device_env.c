// RUN: %libomptarget-compile-generic
// RUN: env OMP_DEFAULT_DEVICE=5 OMP_TARGET_OFFLOAD=disabled
// %libomptarget-run-generic 2>&1 | %fcheck-generic
//
// Integration test: OMP_DEFAULT_DEVICE environment variable is properly
// overridden when offload is disabled

#include <omp.h>
#include <stdio.h>

int main() {
  // Force runtime initialization to parse env vars
#pragma omp parallel
  {
  }

  int initial_device = omp_get_initial_device();
  int default_device = omp_get_default_device();
  int num_devices = omp_get_num_devices();

  // CHECK: num_devices: 0
  printf("num_devices: %d\n", num_devices);

  // CHECK: initial_device: [[INITIAL:[0-9]+]]
  printf("initial_device: %d\n", initial_device);

  // Even though OMP_DEFAULT_DEVICE=5, should get initial_device
  // CHECK: default_device: [[INITIAL]]
  printf("default_device: %d\n", default_device);

  // CHECK: Match: YES
  printf("Match: %s\n", (default_device == initial_device) ? "YES" : "NO");

  // Verify target region executes on host
  int is_host = 0;
#pragma omp target map(from : is_host)
  {
    is_host = omp_is_initial_device();
  }

  // CHECK: Target on host: YES
  printf("Target on host: %s\n", is_host ? "YES" : "NO");

  return 0;
}
