// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

int main()
{
  int x = 0;
  #pragma omp parallel num_threads(3)
  {
    #pragma omp atomic
    x++;
  }
  #pragma omp parallel num_threads(2)
  {
    #pragma omp atomic
    x++;
  }


  printf("x=%d\n", x);

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_idle'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_idle_begin:
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_idle_end:

  return 0;
}
