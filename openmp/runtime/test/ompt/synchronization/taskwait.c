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
#pragma omp master
    {
#pragma omp task
      {
        x++;
      }
#pragma omp taskwait
      print_current_address(1);
    }
  }

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region_wait'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_taskwait_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[RETURN_ADDRESS:(0x)?[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_taskwait_begin: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_taskwait_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: ompt_event_taskwait_end: parallel_id={{[0-f]+}}, task_id={{[0-f]+}}, codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: current_address={{.*}}[[RETURN_ADDRESS]]
  // clang-format on

  return 0;
}
