// RUN: %libomp-compile-and-run
// REQUIRES: ompt
//
// Test that omp_get_default_device() returns the initial device (0) when
// called from within a target region when OMP_TARGET_OFFLOAD=DISABLED.
// The target region should execute on the host.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void kmp_set_defaults(char const *str);

int main() {
  // Set non-zero default device using API (more direct than env var)
  omp_set_default_device(4);

  // Now disable offload
  kmp_set_defaults("OMP_TARGET_OFFLOAD=DISABLED");

// Force parallel region to initialize runtime
#pragma omp parallel
  {
  }

  int initial_device = omp_get_initial_device();
  int host_default_device = omp_get_default_device();
  int target_default_device = -1;
  int target_is_initial = -1;

  printf("Host context:\n");
  printf("  initial_device = %d\n", initial_device);
  printf("  default_device = %d\n", host_default_device);

// Call omp_get_default_device() from within target region
// When offload is disabled, this should execute on host
#pragma omp target map(from : target_default_device, target_is_initial)
  {
    target_default_device = omp_get_default_device();
    target_is_initial = omp_is_initial_device();
  }

  printf("Target context (executed on host when offload disabled):\n");
  printf("  default_device = %d\n", target_default_device);
  printf("  is_initial_device = %d\n", target_is_initial);

  // When offload is disabled, target region executes on host
  if (target_is_initial != 1) {
    fprintf(stderr, "FAIL: Target region did not execute on initial device\n");
    return EXIT_FAILURE;
  }

  // Both host and target context should return same device
  if (host_default_device != initial_device) {
    fprintf(stderr, "FAIL: Host default_device (%d) != initial_device (%d)\n",
            host_default_device, initial_device);
    return EXIT_FAILURE;
  }

  if (target_default_device != initial_device) {
    fprintf(stderr, "FAIL: Target default_device (%d) != initial_device (%d)\n",
            target_default_device, initial_device);
    return EXIT_FAILURE;
  }

  printf("PASS: default_device returns initial_device in both host and target "
         "contexts\n");
  return EXIT_SUCCESS;
}
