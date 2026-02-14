// RUN: %libomp-compile-and-run
//
// Test that omp_set_default_device() stores initial_device in the ICV
// when OMP_TARGET_OFFLOAD=DISABLED, making omp_get_default_device() return
// a valid value.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void kmp_set_defaults(char const *str);

int main() {
  // Disable offload first
  kmp_set_defaults("OMP_TARGET_OFFLOAD=DISABLED");

  // Force runtime initialization
#pragma omp parallel
  {
  }

  int initial_device = omp_get_initial_device();
  int num_devices = omp_get_num_devices();

  printf("Configuration:\n");
  printf("  num_devices = %d\n", num_devices);
  printf("  initial_device = %d\n", initial_device);

  // Test: Set default device to invalid value
  omp_set_default_device(5);

  // Get default device - should return initial_device (not 5)
  // because setter clamped it
  int default_device = omp_get_default_device();

  printf("After omp_set_default_device(5):\n");
  printf("  default_device = %d\n", default_device);

  if (default_device != initial_device) {
    fprintf(stderr, "FAIL: Setter didn't clamp to initial_device\n");
    fprintf(stderr, "      Expected %d, got %d\n", initial_device,
            default_device);
    return EXIT_FAILURE;
  }

  // Try another value
  omp_set_default_device(10);
  default_device = omp_get_default_device();

  printf("After omp_set_default_device(10):\n");
  printf("  default_device = %d\n", default_device);

  if (default_device != initial_device) {
    fprintf(stderr, "FAIL: Setter didn't clamp second value\n");
    return EXIT_FAILURE;
  }

  // Verify target region works
  int executed = 0;
#pragma omp target map(tofrom : executed)
  {
    executed = 1;
  }

  if (!executed) {
    fprintf(stderr, "FAIL: Target region didn't execute\n");
    return EXIT_FAILURE;
  }

  printf("PASS: Setter correctly clamps invalid device to initial_device\n");
  return EXIT_SUCCESS;
}
