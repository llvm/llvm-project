// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=disabled %libomptarget-run-generic 2>&1 |
// %fcheck-generic
//
// Test omp_get_default_device() API behavior when offload is disabled

#include <omp.h>
#include <stdio.h>

int main() {
  // Test 1: Default behavior
  int dev1 = omp_get_default_device();
  // CHECK: Test 1: {{0}}
  printf("Test 1: %d\n", dev1);

  // Test 2: After setting device
  omp_set_default_device(3);
  int dev2 = omp_get_default_device();
  // CHECK: Test 2: {{0}}
  printf("Test 2: %d\n", dev2);

  // Test 3: Multiple sets
  for (int i = 0; i < 5; i++) {
    omp_set_default_device(i + 10);
    int dev = omp_get_default_device();
    // CHECK: Test 3.{{[0-4]}}: {{0}}
    printf("Test 3.%d: %d\n", i, dev);
  }

  // Test 4: Consistency with initial device
  int initial = omp_get_initial_device();
  int default_dev = omp_get_default_device();
  // CHECK: Test 4: EQUAL
  printf("Test 4: %s\n", (initial == default_dev) ? "EQUAL" : "NOT_EQUAL");

  return 0;
}
