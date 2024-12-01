// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
// XFAIL: gcc-4

#include "callback.h"
#include <omp.h>

int main()
{
  int y[] = {0,1,2,3};

  int i;
  #pragma omp for simd
  for (i = 0; i < 4; i++)
  {
    y[i]++;
  }

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region_wait'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // master thread implicit barrier at simd loop end
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_barrier_implicit_workshare_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_workshare_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_workshare_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_workshare_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // clang-format on

  return 0;
}
