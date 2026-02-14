// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=disabled %libomptarget-run-generic 2>&1 |
// %fcheck-generic
//
// API test: omp_set_default_device() should clamp to initial_device
// when OMP_TARGET_OFFLOAD=disabled

#include <omp.h>
#include <stdio.h>

int main() {
  // Force runtime initialization
#pragma omp parallel
  {
  }

  int initial = omp_get_initial_device();
  int num_devices = omp_get_num_devices();

  // CHECK: num_devices: 0
  printf("num_devices: %d\n", num_devices);
  // CHECK: initial_device: [[INITIAL:[0-9]+]]
  printf("initial_device: %d\n", initial);

  // Test 1: Set to high value
  omp_set_default_device(10);
  int dev1 = omp_get_default_device();
  // CHECK: After set(10): [[INITIAL]]
  printf("After set(10): %d\n", dev1);

  // Test 2: Set to another high value
  omp_set_default_device(5);
  int dev2 = omp_get_default_device();
  // CHECK: After set(5): [[INITIAL]]
  printf("After set(5): %d\n", dev2);

  // Test 3: All should be equal to initial
  // CHECK: All equal: YES
  printf("All equal: %s\n",
         (dev1 == initial && dev2 == initial) ? "YES" : "NO");

  // Test 4: Target region should work
  int executed = 0;
#pragma omp target map(tofrom : executed)
  {
    executed = 1;
  }

  // CHECK: Target executed: YES
  printf("Target executed: %s\n", executed ? "YES" : "NO");

  return 0;
}
