// RUN: %libomp-compile && env OMP_CANCELLATION=true %libomp-run

// A cancelled untied task is never invoked, but it has already been made the
// thread's current task by __kmp_task_start(), and it still runs
// __kmp_task_finish() -> __kmp_release_deps() to release its successors while
// it is current. So it needs td_last_tied initialized just like a task
// that actually executes.

#include <stdio.h>
#include <omp.h>

#define REPS 10
#define NDEPS 256

int x[NDEPS];

int main(void) {
  if (!omp_get_cancellation()) {
    fprintf(stderr, "cancellation disabled, set OMP_CANCELLATION=true\n");
    return 1;
  }

  for (int r = 0; r < REPS; ++r) {
#pragma omp parallel
    {
#pragma omp single
      {
#pragma omp taskgroup
        {
          // Request cancellation up front so the tasks created below are
          // discarded rather than executed.
#pragma omp task
          {
#pragma omp cancel taskgroup
          }
          for (int i = 0; i < NDEPS; ++i) {
            // An untied predecessor with a tied successor: the tied one is the
            // candidate that makes the constraint check read td_last_tied off
            // the predecessor while it is releasing its dependences.
#pragma omp task untied depend(out : x[i])
            {
              x[i]++;
            }
#pragma omp task depend(in : x[i])
            {
              x[i]++;
            }
          }
        }
      }
    }
  }

  // How many tasks ran is not deterministic under cancellation, so there is
  // nothing to verify beyond completing without crashing or hanging.
  printf("passed\n");
  return 0;
}
