// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// GCC generates code that does not call the runtime for the single construct
// XFAIL: gcc

#include "callback.h"
#include <omp.h>

int main()
{
  int x = 0;
  #pragma omp parallel num_threads(2)
  {
    #pragma omp single
    {
      x++;
    }
  }

  printf("x=%d\n", x);

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_work'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[THREAD_ID_1:[0-9]+]]: ompt_event_single_in_block_begin: parallel_id=[[PARALLEL_ID:[0-9]+]], parent_task_id=[[TASK_ID:[0-9]+]], codeptr_ra={{0x[0-f]+}}, count=1
  // CHECK: {{^}}[[THREAD_ID_1]]: ompt_event_single_in_block_end: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]], codeptr_ra={{0x[0-f]+}}, count=1

  // CHECK: {{^}}[[THREAD_ID_2:[0-9]+]]: ompt_event_single_others_begin: parallel_id=[[PARALLEL_ID:[0-9]+]], task_id=[[TASK_ID:[0-9]+]], codeptr_ra={{0x[0-f]+}}, count=1
  // CHECK: {{^}}[[THREAD_ID_2]]: ompt_event_single_others_end: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]], codeptr_ra={{0x[0-f]+}}, count=1



  return 0;
}
