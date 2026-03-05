// RUN: %libomp-compile-and-run
//
// Test that omp_get_default_device() returns the initial device when
// OMP_TARGET_OFFLOAD=DISABLED, and that target regions execute on the host.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void kmp_set_defaults(char const *str);

int main() {
  // Disable offload first to avoid early runtime initialization
  kmp_set_defaults("OMP_TARGET_OFFLOAD=DISABLED");

  // Set non-zero default device using API (more direct than env var)
  omp_set_default_device(4);

// Force parallel region to initialize runtime
#pragma omp parallel
  {
  }

  int initial_device = omp_get_initial_device();
  int host_default_device = omp_get_default_device();
  int target_is_initial = -1;

  printf("Host context:\n");
  printf("  initial_device = %d\n", initial_device);
  printf("  default_device = %d\n", host_default_device);

  // Verify default_device returns initial_device in host context
  if (host_default_device != initial_device) {
    fprintf(stderr, "FAIL: Host default_device (%d) != initial_device (%d)\n",
            host_default_device, initial_device);
    return EXIT_FAILURE;
  }

  // Verify target region executes on host when offload is disabled
#pragma omp target map(from : target_is_initial)
  {
    target_is_initial = omp_is_initial_device();
  }

  printf("Target region:\n");
  printf("  is_initial_device = %d\n", target_is_initial);

  if (target_is_initial != 1) {
    fprintf(stderr, "FAIL: Target region did not execute on initial device\n");
    return EXIT_FAILURE;
  }

  printf("PASS: default_device returns initial_device and target executes on "
         "host\n");
  return EXIT_SUCCESS;
}
