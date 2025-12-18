// RUN: %libomp-compile-and-run
//
// Test that device operations using omp_get_default_device() don't crash
// when OMP_TARGET_OFFLOAD=DISABLED. This simulates real-world usage where
// the default device is used for device-specific operations.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void kmp_set_defaults(char const *str);

int main() {
  // Disable offload first to avoid early runtime initialization
  kmp_set_defaults("OMP_TARGET_OFFLOAD=DISABLED");

  // Simulate the problematic scenario: high default device number + disabled
  // offload Use API call instead of env var to ensure ICV is set
  omp_set_default_device(10);

// Force parallel region to initialize runtime
#pragma omp parallel
  {
  }

  int device = omp_get_default_device();
  int num_devices = omp_get_num_devices();
  int initial_device = omp_get_initial_device();

  printf("Configuration:\n");
  printf("  num_devices = %d\n", num_devices);
  printf("  initial_device = %d\n", initial_device);
  printf("  default_device = %d\n", device);

  // Verify device is in valid range
  if (device < 0 || device > num_devices) {
    fprintf(stderr, "FAIL: default_device (%d) is out of valid range [0, %d]\n",
            device, num_devices);
    return EXIT_FAILURE;
  }

  // Test 1: Check if we're on initial device (should be true when offload
  // disabled)
  int is_initial = (device == initial_device);
  if (!is_initial) {
    fprintf(stderr,
            "FAIL: default_device (%d) is not the initial_device (%d)\n",
            device, initial_device);
    return EXIT_FAILURE;
  }

  // Test 2: Use device in target region with device clause
  // This should not crash even though OMP_DEFAULT_DEVICE=10
  int result = -1;
#pragma omp target device(device) map(from : result)
  {
    result = omp_get_device_num();
  }

  printf("Target region executed on device: %d\n", result);

  // Test 3: Query device properties using the default device
  int is_host = (device == initial_device);
  printf("Device %d is_host: %d\n", device, is_host);

  // Test 4: Verify target region with default device specification
  int test_value = 0;
#pragma omp target device(omp_get_default_device()) map(tofrom : test_value)
  {
    test_value = 42;
  }

  if (test_value != 42) {
    fprintf(stderr,
            "FAIL: Target region with default device did not execute\n");
    return EXIT_FAILURE;
  }

  printf("PASS: All device operations completed without crash\n");
  return EXIT_SUCCESS;
}
