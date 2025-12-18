// RUN: %libomp-compile-and-run
//
// Test that omp_get_default_device() returns the initial device (0) when
// OMP_TARGET_OFFLOAD=DISABLED, with both OMP_DEFAULT_DEVICE environment
// variable and omp_set_default_device() API call setting non-zero values.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void kmp_set_defaults(char const *str);

int main() {
  // Simulate worst case: both environment variable and API call set non-zero
  // device
  kmp_set_defaults("OMP_DEFAULT_DEVICE=3");
  kmp_set_defaults("OMP_TARGET_OFFLOAD=DISABLED");

// Force parallel region to initialize runtime
#pragma omp parallel
  {
  }

  int initial_device = omp_get_initial_device();
  int default_device_1 = omp_get_default_device();

  printf("With OMP_DEFAULT_DEVICE=3 and OMP_TARGET_OFFLOAD=DISABLED:\n");
  printf("  initial_device = %d\n", initial_device);
  printf("  default_device = %d\n", default_device_1);

  if (default_device_1 != initial_device) {
    fprintf(stderr,
            "FAIL: Environment variable not overridden by offload disabled\n");
    return EXIT_FAILURE;
  }

  // Now also call omp_set_default_device()
  omp_set_default_device(7);
  int default_device_2 = omp_get_default_device();

  printf("After additional omp_set_default_device(7):\n");
  printf("  default_device = %d\n", default_device_2);

  if (default_device_2 != initial_device) {
    fprintf(stderr, "FAIL: API call not overridden by offload disabled\n");
    return EXIT_FAILURE;
  }

  // Verify consistency across multiple calls
  for (int i = 0; i < 5; i++) {
    int dev = omp_get_default_device();
    if (dev != initial_device) {
      fprintf(stderr,
              "FAIL: Inconsistent result on call %d: got %d, expected %d\n", i,
              dev, initial_device);
      return EXIT_FAILURE;
    }
  }

  printf("PASS: default_device consistently returns initial_device\n");
  return EXIT_SUCCESS;
}
