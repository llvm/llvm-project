// RUN: %libomp-compile && %libomp-run | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

int main() {
  unsigned int i, j, x;

#pragma omp parallel num_threads(2)
#pragma omp master
#pragma omp taskloop
  for (j = 0; j < 5; j += 3) {
    x++;
  }

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin:
  // CHECK-SAME: parent_task_id={{[0-9]+}}
  // CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]]
  // CHECK-SAME: requested_team_size=2
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: task_id=[[IMPLICIT_TASK_ID1:[0-9]+]]
  // CHECK-SAME: team_size=2, thread_num=0
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskgroup_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID1]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskloop_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: parent_task_id=[[IMPLICIT_TASK_ID1]]
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]], count=2
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
  // CHECK-SAME: parent_task_id=[[IMPLICIT_TASK_ID1]]
  // CHECK-SAME: new_task_id=[[TASK_ID1:[0-9]+]]
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK-SAME: task_type=ompt_task_explicit=4
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
  // CHECK-SAME: parent_task_id=[[IMPLICIT_TASK_ID1]]
  // CHECK-SAME: new_task_id=[[TASK_ID2:[0-9]+]]
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK-SAME: task_type=ompt_task_explicit=4
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskloop_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: parent_task_id=[[IMPLICIT_TASK_ID1]]
  // CHECK-SAME: count=2
  // CHECK-DAG: {{^}}[[MASTER_ID]]: ompt_event_wait_taskgroup_begin:
  // Schedule events:
  // CHECK-DAG: {{^.*}}first_task_id={{[0-9]+}}, second_task_id=[[TASK_ID1]]
  // CHECK-DAG: {{^.*}}first_task_id=[[TASK_ID1]], second_task_id={{[0-9]+}}
  // CHECK-DAG: {{^.*}}first_task_id={{[0-9]+}}, second_task_id=[[TASK_ID2]]
  // CHECK-DAG: {{^.*}}first_task_id=[[TASK_ID2]], second_task_id={{[0-9]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_taskgroup_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID1]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskgroup_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID1]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id=0
  // CHECK-SAME: task_id=[[IMPLICIT_TASK_ID1]], team_size=2, thread_num=0
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]]

  return 0;
}
