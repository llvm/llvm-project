// RUN: %libomp-compile-and-run
//
// Test that omp_get_default_device() returns the initial device consistently
// across nested parallel regions and with ICV inheritance when
// OMP_TARGET_OFFLOAD=DISABLED.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void kmp_set_defaults(char const *str);

int check_default_device(const char *context) {
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

  // Set configuration
  kmp_set_defaults("OMP_DEFAULT_DEVICE=8");
  kmp_set_defaults("OMP_TARGET_OFFLOAD=DISABLED");

// Initialize runtime
#pragma omp parallel
  {
  }

  int initial_device = omp_get_initial_device();
  printf("initial_device = %d\n", initial_device);

  // Test 1: Sequential region
  errors += check_default_device("sequential");

// Test 2: Parallel region
#pragma omp parallel reduction(+ : errors)
  {
    errors += check_default_device("parallel");

// Test 3: Nested parallel (if supported)
#pragma omp parallel reduction(+ : errors) if (omp_get_max_threads() > 2)
    {
      errors += check_default_device("nested parallel");
    }
  }

// Test 4: After modifying in one thread
#pragma omp parallel num_threads(4) reduction(+ : errors)
  {
    int tid = omp_get_thread_num();

    // Each thread tries to set different default device
    omp_set_default_device(tid + 20);

    // But should still get initial device
    errors += check_default_device("after thread-local set");

#pragma omp barrier

    // Check again after barrier
    errors += check_default_device("after barrier");
  }

  // Test 5: Back in sequential after all the parallel regions
  errors += check_default_device("sequential final");

  // Test 6: Target region context
  int target_errors = 0;
#pragma omp target map(tofrom : target_errors)
  {
    int default_dev = omp_get_default_device();
    int initial_dev = omp_get_initial_device();
    if (default_dev != initial_dev) {
      target_errors = 1;
    }
  }
  errors += target_errors;

  if (errors > 0) {
    fprintf(stderr, "FAIL: %d errors detected\n", errors);
    return EXIT_FAILURE;
  }

  printf("PASS: default_device consistently returns initial_device across all "
         "contexts\n");
  return EXIT_SUCCESS;
}
