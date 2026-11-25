// RUN: %libomp-compile-and-run

// Tests that an ordered loop with schedule(runtime) when the run-time
// schedule is set to static_steal with omp_set_schedule(), does not
// use the unordered static-steal scheduling nor hang.
// Such loops must fall back to dynamic, as they do with the environment
// variable OMP_SCHEDULE=static_steal.

#include <omp.h>
#include "omp_testsuite.h"

// ---------------------------------------------------------------------------
// Definition copied from OpenMP RTL (kmp_sched_t in kmp.h).
enum { kmp_sched_static_steal = 102 };
// End of definition copied from OpenMP RTL.
// ---------------------------------------------------------------------------

static int last_i = 0;

/* Utility function to check that i is increasing monotonically
   with each call */
static int check_i_islarger(int i) {
  int islarger;
  islarger = (i > last_i);
  last_i = i;
  return (islarger);
}

int test_static_steal_ordered() {
  int sum;
  int is_larger = 1;
  int known_sum;
  int i;

  last_i = 0;
  sum = 0;
  omp_set_schedule((omp_sched_t)kmp_sched_static_steal, 4);

  // num_threads(4): static_steal needs more than one thread.
#pragma omp parallel for schedule(runtime) ordered num_threads(4)
  for (i = 1; i <= LOOPCOUNT; i++) {
#pragma omp ordered
    {
      is_larger = check_i_islarger(i) && is_larger;
      sum = sum + i;
    }
  }

  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
  return (known_sum == sum) && is_larger;
}

int main() {
  int i;
  int num_failed = 0;

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_static_steal_ordered()) {
      num_failed++;
    }
  }
  return num_failed;
}