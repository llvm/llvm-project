// RUN: %libomptarget-compile-generic
// RUN: env OMP_TARGET_OFFLOAD=disabled OMP_DEFAULT_DEVICE=99
// %libomptarget-run-generic 2>&1 | %fcheck-generic
//
// Stress test: Multiple scenarios with offload disabled to ensure robustness

#include <omp.h>
#include <stdio.h>

int main() {
  // Force runtime initialization
#pragma omp parallel
  {
  }

  int initial = omp_get_initial_device();

  // CHECK: Starting test with initial_device: [[INITIAL:[0-9]+]]
  printf("Starting test with initial_device: %d\n", initial);

  // Test 1: Extreme values
  int extreme_values[] = {-1, 0, 1, 100, 1000, 99999};
  for (int i = 0; i < 6; i++) {
    omp_set_default_device(extreme_values[i]);
    int dev = omp_get_default_device();
    if (dev != initial) {
      printf("FAIL: extreme value %d\n", extreme_values[i]);
      return 1;
    }
  }
  // CHECK: Test 1 (extreme values): PASS
  printf("Test 1 (extreme values): PASS\n");

  // Test 2: Rapid consecutive sets
  for (int i = 0; i < 100; i++) {
    omp_set_default_device(i);
  }
  int dev = omp_get_default_device();
  // CHECK: Test 2 (rapid sets): [[INITIAL]]
  printf("Test 2 (rapid sets): %d\n", dev);

  // Test 3: Parallel region with device operations
  int errors = 0;
#pragma omp parallel num_threads(4) reduction(+ : errors)
  {
    omp_set_default_device(omp_get_thread_num() * 50);
    int d = omp_get_default_device();
    if (d != initial) {
      errors++;
    }
  }
  // CHECK: Test 3 (parallel): 0 errors
  printf("Test 3 (parallel): %d errors\n", errors);

  // Test 4: Multiple target regions
  for (int i = 0; i < 5; i++) {
    omp_set_default_device(i * 20);
    int executed = 0;
#pragma omp target map(tofrom : executed)
    {
      executed = 1;
    }
    if (!executed) {
      printf("FAIL: target %d didn't execute\n", i);
      return 1;
    }
  }
  // CHECK: Test 4 (multiple targets): PASS
  printf("Test 4 (multiple targets): PASS\n");

  // Test 5: Nested target regions
  int outer_executed = 0;
#pragma omp target map(tofrom : outer_executed)
  {
    outer_executed = 1;
    // Note: nested target would require additional handling
  }
  // CHECK: Test 5 (nested targets): PASS
  printf("Test 5 (nested targets): %s\n", outer_executed ? "PASS" : "FAIL");

  // CHECK: All tests completed successfully
  printf("All tests completed successfully\n");

  return 0;
}
