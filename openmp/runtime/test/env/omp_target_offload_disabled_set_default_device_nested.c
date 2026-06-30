// RUN: %libomp-compile-and-run
//
// Test setter behavior across nested parallel regions and ICV inheritance
// when OMP_TARGET_OFFLOAD=DISABLED

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void kmp_set_defaults(char const *str);

int check_device(const char *context) {
  int default_dev = omp_get_default_device();
  int initial_dev = omp_get_initial_device();

  if (default_dev != initial_dev) {
    fprintf(stderr, "FAIL [%s]: default=%d, initial=%d\n", context, default_dev,
            initial_dev);
    return 1;
  }
  return 0;
}

int main() {
  int errors = 0;

  kmp_set_defaults("OMP_DEFAULT_DEVICE=6");
  kmp_set_defaults("OMP_TARGET_OFFLOAD=DISABLED");

  // Initialize runtime
#pragma omp parallel
  {
  }

  int initial_device = omp_get_initial_device();
  printf("initial_device = %d\n", initial_device);

  // Test 1: Sequential
  errors += check_device("sequential");

  // Test 2: Parallel region
#pragma omp parallel reduction(+ : errors)
  {
    errors += check_device("parallel");

    // Each thread tries to set device
    omp_set_default_device(omp_get_thread_num() + 20);
    errors += check_device("after thread set");

    // Test 3: Nested parallel
#pragma omp parallel reduction(+ : errors) if (omp_get_max_threads() > 2)
    {
      errors += check_device("nested parallel");

      // Nested thread also tries to set
      omp_set_default_device(omp_get_thread_num() + 50);
      errors += check_device("nested after set");
    }

#pragma omp barrier
    errors += check_device("after barrier");
  }

  // Test 4: Back in sequential
  errors += check_device("sequential final");

  // Test 5: Target region execution (only use well-defined functions)
  // Note: Calling omp_get_default_device() from within target region is
  // unspecified
  int target_errors = 0;
#pragma omp target map(tofrom : target_errors)
  {
    // Verify target executes on initial device when offload disabled
    if (!omp_is_initial_device()) {
      target_errors = 1;
    }
  }
  errors += target_errors;

  if (errors > 0) {
    fprintf(stderr, "FAIL: %d errors detected\n", errors);
    return EXIT_FAILURE;
  }

  printf("PASS: Consistent behavior across all nesting levels\n");
  return EXIT_SUCCESS;
}
