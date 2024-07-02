#include "callback.h"
#include <omp.h>
#include <stdio.h>

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#ifndef SCHED_OUTPUT
#define SCHED_OUTPUT STR(SCHEDULE)
#endif

int main()
{
  unsigned int i;
  printf("0: Schedule: " SCHED_OUTPUT "\n");

#pragma omp parallel for num_threads(4) schedule(SCHEDULE)
  for (i = 0; i < 64; i++) {
  }

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_end'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_work'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: 0: Schedule: [[SCHED:[a-z]+]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin: parent_task_id={{[0-9]+}}, parent_task_frame.exit=[[NULL]], parent_task_frame.reenter={{0x[0-f]+}}, parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=4, codeptr_ra=0x{{[0-f]+}}, invoker={{[0-9]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_loop_[[SCHED]]_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], codeptr_ra=
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_loop_[[SCHED]]_end: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_loop_[[SCHED]]_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], codeptr_ra=
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_loop_[[SCHED]]_end: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_loop_[[SCHED]]_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], codeptr_ra=
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_loop_[[SCHED]]_end: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_loop_[[SCHED]]_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], codeptr_ra=
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_loop_[[SCHED]]_end: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // clang-format on

  return 0;
}
