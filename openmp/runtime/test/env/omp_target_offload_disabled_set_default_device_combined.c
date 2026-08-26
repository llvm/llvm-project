// RUN: %libomp-compile-and-run
//
// Test combined scenario: OMP_DEFAULT_DEVICE env var + omp_set_default_device()
// API Both should be overridden when OMP_TARGET_OFFLOAD=DISABLED

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void kmp_set_defaults(char const *str);

int main() {
  // Worst case: both env var and API try to set invalid device
  kmp_set_defaults("OMP_DEFAULT_DEVICE=3");
  kmp_set_defaults("OMP_TARGET_OFFLOAD=DISABLED");

  // Force runtime initialization
#pragma omp parallel
  {
  }

  int initial_device = omp_get_initial_device();
  int default_device = omp_get_default_device();

  printf("Test 1: With OMP_DEFAULT_DEVICE=3 and offload disabled\n");
  printf("  initial_device = %d\n", initial_device);
  printf("  default_device = %d\n", default_device);

  if (default_device != initial_device) {
    fprintf(stderr, "FAIL: Env var not overridden\n");
    return EXIT_FAILURE;
  }

  // Now try API call on top of env var
  omp_set_default_device(7);
  default_device = omp_get_default_device();

  printf("\nTest 2: After additional omp_set_default_device(7)\n");
  printf("  default_device = %d\n", default_device);

  if (default_device != initial_device) {
    fprintf(stderr, "FAIL: API call not overridden\n");
    return EXIT_FAILURE;
  }

  // Test 3: Multiple sets in sequence
  for (int i = 0; i < 5; i++) {
    omp_set_default_device(i + 10);
    default_device = omp_get_default_device();
    if (default_device != initial_device) {
      fprintf(stderr, "FAIL: Set %d not overridden\n", i);
      return EXIT_FAILURE;
    }
  }

  // Test 4: Use in target region
  int result = -1;
#pragma omp target device(omp_get_default_device()) map(from : result)
  {
    result = omp_is_initial_device();
  }

  if (result != 1) {
    fprintf(stderr, "FAIL: Target didn't execute on initial device\n");
    return EXIT_FAILURE;
  }

  printf("\nPASS: All combinations correctly override invalid device\n");
  return EXIT_SUCCESS;
}
