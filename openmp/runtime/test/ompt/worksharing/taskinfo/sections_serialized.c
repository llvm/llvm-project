// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// clang-format on

#include "callback.h"
#include <omp.h>

int main() {
#pragma omp parallel sections num_threads(1)
  {
#pragma omp section
    {
      // implicit task info
      print_ids(0);
      // initial task info
      print_ids(1);
    }
  }

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'


  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_initial_task_begin: parallel_id=[[INITIAL_PARALLEL_ID:[0-f]+]], task_id=[[INITIAL_TASK_ID:[0-f]+]], actual_parallelism=1, index=1, flags=1

  // region 0
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin
  // CHECK-SAME: parent_task_frame.exit=[[NULL]], parent_task_frame.reenter=[[INITIAL_TASK_FRAME_ENTER:(0x)?[0-f]+]],
  // CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-f]+]]

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID:[0-f]+]]

  // information about implicit task (exit frame should be set, while enter should be NULL)
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]]
  // CHECK-SAME: exit_frame={{(0x)?[0-f]+}}
  // CHECK-SAME: reenter_frame=[[NULL]]
  // CHECK-SAME: task_type=ompt_task_implicit

  // information about initial task (exit frame should be NULL, while enter frame shoule be set)
  // CHECK: {{^}}[[MASTER_ID]]: task level 1: parallel_id=[[INITIAL_PARALLEL_ID]], task_id=[[INITIAL_TASK_ID]]
  // CHECK-SAME: exit_frame=[[NULL]]
  // CHECK-SAME: reenter_frame=[[INITIAL_TASK_FRAME_ENTER]]
  // CHECK-SAME: task_type=ompt_task_initial
  // clang-format on

  return 0;
}
