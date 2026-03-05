// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=disabled %libomptarget-run-generic 2>&1 |
// %fcheck-generic
//
// API contract test: omp_get_default_device() behavior when offload is disabled

#include <omp.h>
#include <stdio.h>

int main() {
  // Force runtime initialization to parse environment variables
#pragma omp parallel
  {
  }

  int initial = omp_get_initial_device();

  // Test 1: Default device should initially equal initial device
  int dev1 = omp_get_default_device();
  // CHECK: Test 1: EQUAL
  printf("Test 1: %s\n", (dev1 == initial) ? "EQUAL" : "NOT_EQUAL");

  // Test 2: After setting to 3, get should still return initial device (not 3)
  omp_set_default_device(3);
  int dev2 = omp_get_default_device();
  // CHECK: Test 2: EQUAL
  printf("Test 2: %s\n", (dev2 == initial) ? "EQUAL" : "NOT_EQUAL");

  // Test 3: After setting to 10, get should still return initial device
  omp_set_default_device(10);
  int dev3 = omp_get_default_device();
  // CHECK: Test 3: EQUAL
  printf("Test 3: %s\n", (dev3 == initial) ? "EQUAL" : "NOT_EQUAL");

  // Test 4: All calls return consistent value
  // CHECK: Test 4: CONSISTENT
  printf("Test 4: %s\n",
         (dev1 == dev2 && dev2 == dev3) ? "CONSISTENT" : "INCONSISTENT");

  return 0;
}
