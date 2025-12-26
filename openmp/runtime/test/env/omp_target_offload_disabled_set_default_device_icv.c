// RUN: %libomp-compile-and-run
//
// Test that the ICV (Internal Control Variable) itself contains the correct
// value This verifies spec compliance: omp_get_default_device() returns the ICV
// value, and the setter ensures the ICV is always valid.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void kmp_set_defaults(char const *str);

int main() {
  kmp_set_defaults("OMP_TARGET_OFFLOAD=DISABLED");

  // Force runtime initialization
#pragma omp parallel
  {
  }

  int initial_device = omp_get_initial_device();
  printf("initial_device = %d\n", initial_device);

  // Test 1: Initial value should be initial_device
  int dev1 = omp_get_default_device();
  printf("Test 1 - Initial: default_device = %d\n", dev1);
  if (dev1 != initial_device) {
    fprintf(stderr, "FAIL: Initial default != initial_device\n");
    return EXIT_FAILURE;
  }

  // Test 2: After setting to 5, getter should still return initial_device
  // This proves the ICV was set to initial_device (not 5)
  omp_set_default_device(5);
  int dev2 = omp_get_default_device();
  printf("Test 2 - After set(5): default_device = %d\n", dev2);
  if (dev2 != initial_device) {
    fprintf(stderr, "FAIL: Setter didn't update ICV to initial_device\n");
    return EXIT_FAILURE;
  }

  // Test 3: Multiple sets should all result in initial_device
  int test_values[] = {10, 20, 100, -1, 999};
  for (int i = 0; i < 5; i++) {
    omp_set_default_device(test_values[i]);
    int dev = omp_get_default_device();
    printf("Test 3.%d - After set(%d): default_device = %d\n", i,
           test_values[i], dev);
    if (dev != initial_device) {
      fprintf(stderr, "FAIL: Set(%d) resulted in %d\n", test_values[i], dev);
      return EXIT_FAILURE;
    }
  }

  // Test 4: Verify getter is truly just returning the ICV
  // Call getter multiple times without any setter calls
  for (int i = 0; i < 10; i++) {
    int dev = omp_get_default_device();
    if (dev != initial_device) {
      fprintf(stderr, "FAIL: Getter call %d returned %d\n", i, dev);
      return EXIT_FAILURE;
    }
  }

  // Test 5: Use the device in actual operation
  int executed = 0;
  int device_for_target = omp_get_default_device();
#pragma omp target device(device_for_target) map(tofrom : executed)
  {
    executed = 1;
  }

  if (!executed) {
    fprintf(stderr, "FAIL: Target with default_device didn't execute\n");
    return EXIT_FAILURE;
  }

  printf("PASS: ICV always contains valid initial_device value\n");
  return EXIT_SUCCESS;
}
