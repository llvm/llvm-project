// clang-format off
// RUN:  %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
// clang-format on

#include "callback.h"
#include <unistd.h>
#include <stdio.h>

int main() {
  int condition = 0;
  int x = 0;
#pragma omp parallel num_threads(2)
  {
#pragma omp master
    {
#pragma omp taskgroup
      {
        print_current_address(1);
#pragma omp task
        {
#pragma omp atomic
          x++;
        }
      }
      print_current_address(2);
    }
  }

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_schedule'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_cancel'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_thread_begin'


  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_taskgroup_begin: parallel_id=[[PARALLEL_ID:[0-f]+]], task_id=[[TASK_ID:[0-f]+]], codeptr_ra=[[RETURN_ADDRESS:(0x)?[0-f]+]]
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: current_address={{.*}}[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_taskgroup_begin: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]], codeptr_ra=[[RETURN_ADDRESS:(0x)?[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_taskgroup_end: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]], codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskgroup_end: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]], codeptr_ra=[[RETURN_ADDRESS]]
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: current_address={{.*}}[[RETURN_ADDRESS]]
  // clang-format on

  return 0;
}
