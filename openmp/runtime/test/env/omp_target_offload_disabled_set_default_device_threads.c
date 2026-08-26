// RUN: %libomp-compile-and-run
//
// Multi-threaded stress test: verify setter correctly clamps device numbers
// across concurrent threads when OMP_TARGET_OFFLOAD=DISABLED

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void kmp_set_defaults(char const *str);

int main() {
  const int NUM_THREADS = 8;
  const int NUM_ITERATIONS = 50;

  kmp_set_defaults("OMP_TARGET_OFFLOAD=DISABLED");

  // Force runtime initialization
#pragma omp parallel
  {
  }

  int initial_device = omp_get_initial_device();
  int errors = 0;

  printf("Multi-threaded test: %d threads, %d iterations each\n", NUM_THREADS,
         NUM_ITERATIONS);
  printf("initial_device = %d\n", initial_device);

#pragma omp parallel num_threads(NUM_THREADS) reduction(+ : errors)
  {
    int tid = omp_get_thread_num();

    for (int i = 0; i < NUM_ITERATIONS; i++) {
      // Each thread tries to set different device
      int attempt_device = tid * 100 + i;
      omp_set_default_device(attempt_device);

      // Should always get initial_device back
      int default_device = omp_get_default_device();

      if (default_device != initial_device) {
#pragma omp critical
        {
          fprintf(stderr,
                  "FAIL: Thread %d iteration %d: set %d, got %d, expected %d\n",
                  tid, i, attempt_device, default_device, initial_device);
        }
        errors++;
      }
    }
  }

  if (errors > 0) {
    fprintf(stderr, "FAIL: %d errors across all threads\n", errors);
    return EXIT_FAILURE;
  }

  // Final verification
  int final_device = omp_get_default_device();
  if (final_device != initial_device) {
    fprintf(stderr, "FAIL: Final check failed\n");
    return EXIT_FAILURE;
  }

  printf("PASS: All %d threads consistently got initial_device\n",
         NUM_THREADS * NUM_ITERATIONS);
  return EXIT_SUCCESS;
}
