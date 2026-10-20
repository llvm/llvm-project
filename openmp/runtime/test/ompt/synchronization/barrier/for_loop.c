// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
// clang-format on
#include "callback.h"
#include <omp.h>

int main() {
  int y[] = {0, 1, 2, 3};

#pragma omp parallel num_threads(2)
  {
    // implicit barrier at end of for loop
    int i;
#pragma omp for
    for (i = 0; i < 4; i++) {
      y[i]++;
    }
    print_current_address();
  }

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region_wait'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // master thread implicit barrier at loop end
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_barrier_implicit_workshare_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_workshare_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_workshare_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_workshare_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}

  // master thread implicit barrier at parallel end
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_parallel_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_parallel_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}

  // worker thread implicit barrier at loop end
  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_barrier_implicit_workshare_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_implicit_workshare_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_implicit_workshare_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_workshare_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}

  // worker thread implicit barrier after parallel
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_implicit_parallel_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_implicit_parallel_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[NULL]]
  // clang-format on

  return 0;
}
