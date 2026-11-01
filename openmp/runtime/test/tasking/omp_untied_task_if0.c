// RUN: %libomp-compile-and-run

// An untied task whose 'if' clause evaluates to false is executed immediately
// via __kmpc_omp_task_begin_if0(), which makes it the thread's current task.
// It must still inherit td_last_tied from the task it suspends, the way the
// deferred path does in __kmp_invoke_task(). Otherwise it becomes the current
// task with a NULL td_last_tied, and applying the task scheduling constraint
// to any task scheduled from it (here, at the nested taskwait) either asserts
// in __kmp_invoke_task() for an untied candidate, or dereferences NULL in
// __kmp_task_is_allowed() for a tied candidate. See issue #172938.

#include <stdio.h>
#include <omp.h>

#define REPS 50
#define CHILDREN 16

// Not a constant, so the 'if' clause is a runtime condition and the compiler
// emits the __kmpc_omp_task_begin_if0() path for the false case.
int cond = 0;

int main(void) {
  int counter = 0;

  for (int r = 0; r < REPS; ++r) {
#pragma omp parallel
    {
#pragma omp single
      {
#pragma omp task if (cond) untied
        {
          for (int i = 0; i < CHILDREN; ++i) {
            // A tied child exercises __kmp_task_is_allowed(); an untied child
            // exercises the inheritance in __kmp_invoke_task().
            if (i % 2) {
#pragma omp task
              {
#pragma omp atomic
                counter++;
              }
            } else {
#pragma omp task untied
              {
#pragma omp atomic
                counter++;
              }
            }
          }
#pragma omp taskwait
        }
      }
    }
  }

  if (counter != REPS * CHILDREN) {
    fprintf(stderr, "failed: counter = %d, expected %d\n", counter,
            REPS * CHILDREN);
    return 1;
  }
  printf("passed\n");
  return 0;
}
