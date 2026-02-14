// RUN: %libomp-compile-and-run
//
// Simple smoke test to verify omp_get_default_device() returns initial device
// when OMP_TARGET_OFFLOAD=DISABLED with OMP_DEFAULT_DEVICE=2.
// This is the C equivalent of the Fortran smoke test.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void kmp_set_defaults(char const *str);

int main() {
  // Disable offload first to avoid early runtime initialization
  kmp_set_defaults("OMP_TARGET_OFFLOAD=DISABLED");

  // Key to reproducing bug: Set default device to non-zero value
  // This ensures the ICV contains a non-zero value
  omp_set_default_device(2);

// Initialize runtime
#pragma omp parallel
  {
  }

  int num_devices = omp_get_num_devices();
  int initial_device = omp_get_initial_device();
  int default_device = omp_get_default_device();

  // Print results
  printf("number of devices %d\n", num_devices);
  printf("initial device %d\n", initial_device);
  printf("default device %d\n", default_device);

  // The key test: default device should equal initial device
  if (initial_device == default_device) {
    printf("PASS\n");
    return EXIT_SUCCESS;
  } else {
    fprintf(stderr, "FAIL: default_device (%d) != initial_device (%d)\n",
            default_device, initial_device);
    fprintf(stderr, "This would cause: device number '%d' out of range\n",
            default_device);
    return EXIT_FAILURE;
  }
}
