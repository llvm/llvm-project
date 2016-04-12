// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
#include "callback.h"

int main()
{
  #pragma omp parallel num_threads(1)
  {
    print_ids(0);
    print_ids(1);
  }

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame=0x{{[0-f]+}}, parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=1, parallel_function=0x{{[0-f]+}}, invoker=[[PARALLEL_INVOKER:.+]]

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: level 1: parallel_id=0, task_id=[[PARENT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end: parallel_id=[[PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]

  return 0;
}
