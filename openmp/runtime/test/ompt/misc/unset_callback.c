// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include "omp_testsuite.h"
#include <omp.h>

int main()
{
  go_parallel_nthreads(1);
  ompt_set_callback(ompt_callback_parallel_begin, NULL);
  go_parallel_nthreads(1);

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_idle'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_parallel_begin:
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_parallel_end:
  // CHECK-NOT: {{^}}[[THREAD_ID]]: ompt_event_parallel_begin:
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_parallel_end:

  return get_exit_value();
}
