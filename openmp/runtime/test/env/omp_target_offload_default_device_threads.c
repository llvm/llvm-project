// RUN: %libomp-compile-and-run
//
// Test that omp_get_default_device() consistently returns the initial device
// across multiple threads when OMP_TARGET_OFFLOAD=DISABLED.
// This ensures thread-safety.

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

extern void kmp_set_defaults(char const *str);

int main() {
  const int NUM_THREADS = 8;
  const int NUM_ITERATIONS = 100;

  // Set non-zero default device and disable offload
  kmp_set_defaults("OMP_DEFAULT_DEVICE=6");
  kmp_set_defaults("OMP_TARGET_OFFLOAD=DISABLED");

// Force parallel region to initialize runtime
#pragma omp parallel
  {
  }

  int initial_device = omp_get_initial_device();
  int errors = 0;

  printf("Testing with %d threads, %d iterations each\n", NUM_THREADS,
         NUM_ITERATIONS);
  printf("initial_device = %d\n", initial_device);

// Test across multiple parallel regions and threads
#pragma omp parallel num_threads(NUM_THREADS) reduction(+ : errors)
  {
    int tid = omp_get_thread_num();

    for (int i = 0; i < NUM_ITERATIONS; i++) {
      int default_device = omp_get_default_device();

      if (default_device != initial_device) {
#pragma omp critical
        {
          fprintf(
              stderr,
              "FAIL: Thread %d iteration %d: default_device=%d, expected=%d\n",
              tid, i, default_device, initial_device);
        }
        errors++;
      }

      // Also test after setting default device in each thread
      if (i % 10 == 0) {
        omp_set_default_device(tid + 10);
        default_device = omp_get_default_device();

        if (default_device != initial_device) {
#pragma omp critical
          {
            fprintf(
                stderr,
                "FAIL: Thread %d after set: default_device=%d, expected=%d\n",
                tid, default_device, initial_device);
          }
          errors++;
        }
      }
    }
  }

  if (errors > 0) {
    fprintf(stderr, "FAIL: %d errors detected across all threads\n", errors);
    return EXIT_FAILURE;
  }

  // Final verification
  int final_device = omp_get_default_device();
  if (final_device != initial_device) {
    fprintf(stderr,
            "FAIL: Final check failed: default_device=%d, expected=%d\n",
            final_device, initial_device);
    return EXIT_FAILURE;
  }

  printf("PASS: default_device consistently returns initial_device across all "
         "threads\n");
  return EXIT_SUCCESS;
}
