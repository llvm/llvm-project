// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
// clang-format on
#include "callback.h"
#include <omp.h>

int main() {
  int x = 0;

#pragma omp parallel num_threads(2)
  {
#pragma omp atomic
    x++;

#pragma omp barrier
    print_current_address();

#pragma omp atomic
    x++;
  }

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region_wait'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // master thread explicit barrier
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_barrier_explicit_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[RETURN_ADDRESS:(0x)?[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_explicit_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_explicit_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_explicit_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: current_address={{.*}}[[RETURN_ADDRESS]]

  // master thread implicit barrier at parallel end
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_parallel_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_parallel_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra={{(0x)?[0-f]+}}


  // worker thread explicit barrier
  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_barrier_explicit_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[RETURN_ADDRESS:(0x)?[0-f]+]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_explicit_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_explicit_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_explicit_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[THREAD_ID]]: current_address={{.*}}[[RETURN_ADDRESS]]

  // worker thread implicit barrier at parallel end
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_implicit_parallel_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_implicit_parallel_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[NULL]]
  // clang-format on

  return 0;
}
