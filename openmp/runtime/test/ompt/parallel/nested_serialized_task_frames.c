// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// clang-format on

#include "callback.h"
#include <omp.h>

int main() {
#pragma omp parallel num_threads(1)
  {
    // region 0
#pragma omp parallel num_threads(1)
    {
      // region 1
#pragma omp parallel num_threads(1)
      {
        // region 2
        // region 2's implicit task
        print_ids(0);
        // region 1's implicit task
        print_ids(1);
        // region 0's implicit task
        print_ids(2);
        // initial task
        print_ids(3);
      }
    }
  }

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'


  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_initial_task_begin: parallel_id=[[INITIAL_PARALLEL_ID:[0-f]+]], task_id=[[INITIAL_TASK_ID:[0-f]+]], actual_parallelism=1, index=1, flags=1

  // region 0
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin
  // CHECK-SAME: parent_task_frame.exit=[[NULL]], parent_task_frame.reenter=[[INITIAL_TASK_FRAME_ENTER:(0x)?[0-f]+]],
  // CHECK-SAME: parallel_id=[[PARALLEL_ID_0:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID_0]], task_id=[[TASK_ID_0:[0-9]+]]

  // region 1
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin
  // CHECK-SAME: parent_task_frame.exit=[[REGION_0_FRAME_EXIT:(0x)?[0-f]+]], parent_task_frame.reenter=[[REGION_0_FRAME_ENTER:(0x)?[0-f]+]],
  // CHECK-SAME: parallel_id=[[PARALLEL_ID_1:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID_1]], task_id=[[TASK_ID_1:[0-9]+]]

  // region 2
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin
  // CHECK-SAME: parent_task_frame.exit=[[REGION_1_FRAME_EXIT:(0x)?[0-f]+]], parent_task_frame.reenter=[[REGION_1_FRAME_ENTER:(0x)?[0-f]+]],
  // CHECK-SAME: parallel_id=[[PARALLEL_ID_2:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID_2]], task_id=[[TASK_ID_2:[0-9]+]]

  // region 2's implicit task information (exit frame should be set, while enter should be NULL)
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID_2]], task_id=[[TASK_ID_2]]
  // CHECK-SAME: exit_frame={{(0x)?[0-f]+}}
  // CHECK-SAME: reenter_frame=[[NULL]]
  // CHECK-SAME: task_type=ompt_task_implicit

  // region 1's implicit task information (both exit and enter frames should be set)
  // CHECK: {{^}}[[MASTER_ID]]: task level 1: parallel_id=[[PARALLEL_ID_1]], task_id=[[TASK_ID_1]]
  // CHECK-SAME: exit_frame=[[REGION_1_FRAME_EXIT]]
  // CHECK-SAME: reenter_frame=[[REGION_1_FRAME_ENTER]]
  // CHECK-SAME: task_type=ompt_task_implicit

  // region 0's implicit task information (both exit and enter frames should be set)
  // CHECK: {{^}}[[MASTER_ID]]: task level 2: parallel_id=[[PARALLEL_ID_0]], task_id=[[TASK_ID_0]]
  // CHECK-SAME: exit_frame=[[REGION_0_FRAME_EXIT]]
  // CHECK-SAME: reenter_frame=[[REGION_0_FRAME_ENTER]]
  // CHECK-SAME: task_type=ompt_task_implicit

  // region 0's initial task information (both exit and enter frames should be set)
  // CHECK: {{^}}[[MASTER_ID]]: task level 3: parallel_id=[[INITIAL_PARALLEL_ID]], task_id=[[INITIAL_TASK_ID]]
  // CHECK-SAME: exit_frame=[[NULL]]
  // CHECK-SAME: reenter_frame=[[INITIAL_TASK_FRAME_ENTER]]
  // CHECK-SAME: task_type=ompt_task_initial
  // clang-format on

  return 0;
}