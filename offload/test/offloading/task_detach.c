// RUN: %libomptarget-compile-run-and-check-generic

// Check that a detachable task can be used in a target region. Tasks on the
// device are executed immediately by the encountering thread, so the task is
// complete when its structured block finishes, whenever the event is fulfilled.

#include <omp.h>
#include <stdio.h>

int main(void) {
  int in_task = 0, after_task = 0;

#pragma omp target map(tofrom : in_task, after_task)
  {
    omp_event_handle_t event;

#pragma omp task detach(event) shared(in_task)
    {
      in_task = 1;
      omp_fulfill_event(event);
    }
#pragma omp taskwait
    after_task = in_task;

    // Fulfilling the event from outside the task is also accepted.
#pragma omp task detach(event) shared(in_task)
    in_task = 2;

    omp_fulfill_event(event);
#pragma omp taskwait
    after_task += in_task;
  }

  // CHECK: in_task = 2, after_task = 3
  printf("in_task = %d, after_task = %d\n", in_task, after_task);
  return 0;
}
