// RUN: %libomp-compile-and-run

// The test checks that every schedule kind accepted by omp_set_schedule(),
// i.e. that is not considered out of range in __kmp_set_schedule in
// kmp_runtime.cpp, can be set, and we can afterwards get the schedule back with
// omp_get_schedule

#include <omp.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// As with kmp.h, static steal is by default enabled, but can be manually
// disabled. If libomp is built with -DKMP_STATIC_STEAL_ENABLED=0, pass the
// same flag to the tests (for example through OPENMP_TEST_FLAGS).
#ifndef KMP_STATIC_STEAL_ENABLED
#define KMP_STATIC_STEAL_ENABLED 1
#endif

// These definitions need to match kmp_sched_t in kmp.h.
#ifndef KMP_SCHED_TYPE_DEFINED
#define KMP_SCHED_TYPE_DEFINED
typedef enum kmp_sched {
  kmp_sched_lower = 0,
  kmp_sched_static = 1,
  kmp_sched_dynamic = 2,
  kmp_sched_guided = 3,
  kmp_sched_auto = 4,
  kmp_sched_upper_std = 5,
  kmp_sched_lower_ext = 100,
  kmp_sched_trapezoidal = 101,
#if KMP_STATIC_STEAL_ENABLED
  kmp_sched_static_steal = 102,
#endif
  kmp_sched_upper,
  kmp_sched_default = kmp_sched_static,
  kmp_sched_monotonic = 0x80000000
} kmp_sched_t;
#endif

// ---------------------------------------------------------------------------

int main() {
  const int chunk = 5;
  int err = 0;
  unsigned k;

  // Visit every kind that __kmp_set_schedule() accepts: the standard kinds
  // between lower and upper_std, and the extension kinds between lower_ext
  // and upper.
  for (k = kmp_sched_lower + 1; k < kmp_sched_upper; ++k) {
    omp_sched_t kind_get;
    int chunk_get;

    if (k >= kmp_sched_upper_std && k <= kmp_sched_lower_ext) {
      continue;
    }

#ifdef DEBUG
    printf("checking kind %u\n", k);
#endif

    omp_set_schedule((omp_sched_t)k, chunk);
    omp_get_schedule(&kind_get, &chunk_get);

    // Check kind and chunk match.
    // The chunk size is ignored for auto, so allow a mismatched chunk size.
    if (kind_get != (omp_sched_t)k ||
        (k != kmp_sched_auto && chunk_get != chunk)) {
      printf("Error: kind %u: schedule: (%d, %d) is not equal to (%u, %d)\n", k,
             (int)kind_get, chunk_get, k, chunk);
      ++err;
    }
  }

  if (err > 0) {
    printf("Failed\n");
    return 1;
  }
  printf("Passed\n");
  return 0;
}
