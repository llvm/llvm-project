// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=disabled %libomptarget-run-generic 2>&1 |
// %fcheck-generic
//
// Integration test: target region execution when offload is disabled
// with default device set to invalid value

#include <omp.h>
#include <stdio.h>

int main() {
  // Force runtime initialization to parse environment variables
#pragma omp parallel
  {
  }

  // Set high default device number that would be invalid
  omp_set_default_device(5);

  int num_devices = omp_get_num_devices();
  int initial_device = omp_get_initial_device();
  int default_device = omp_get_default_device();

  // CHECK: num_devices: 0
  printf("num_devices: %d\n", num_devices);

  printf("initial_device: %d\n", initial_device);
  printf("default_device: %d\n", default_device);

  // The key test: default device must equal initial device when offload
  // disabled CHECK: PASS
  if (default_device == initial_device) {
    printf("PASS\n");
  } else {
    printf("FAIL: default_device=%d, initial_device=%d\n", default_device,
           initial_device);
    return 1;
  }

  // Verify target region executes without crashing when offload is disabled
  int executed = 0;
#pragma omp target map(tofrom : executed)
  {
    executed = 1;
  }

  // CHECK: Target executed
  if (executed) {
    printf("Target executed\n");
    return 0;
  }

  printf("FAIL: Target region did not execute\n");
  return 1;
}
