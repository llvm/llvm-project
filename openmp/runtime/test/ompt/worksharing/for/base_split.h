#include "callback.h"
#include <omp.h>
#include <stdio.h>

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#ifndef SCHED_OUTPUT
#define SCHED_OUTPUT STR(SCHEDULE)
#endif

/* With the combined parallel-for construct (base.h), the return-addresses are hard to compare.
   With the separate parallel and for-nowait construct, the addresses become more predictable,
   but the begin of the for-loop still generates additional code, so the offset of loop-begin 
   to the label is >4 Byte.
*/

int main()
{
  unsigned int i;
  printf("0: Schedule: " SCHED_OUTPUT "\n");

#pragma omp parallel num_threads(4)
  {
    print_current_address(0);
    #pragma omp for schedule(SCHEDULE) nowait
    for (i = 0; i < 64; i++) {
      print_fuzzy_address(1);
    }
    print_fuzzy_address(2);
  }
  print_fuzzy_address(3);

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_end'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_work'


  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: 0: Schedule: [[SCHED:[a-z]+]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin: parent_task_id={{[0-9]+}}, parent_task_frame.exit=[[NULL]], parent_task_frame.reenter={{0x[0-f]+}}, parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=4, codeptr_ra=[[PARALLEL_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}, invoker={{[0-9]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_loop_[[SCHED]]_begin: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}, codeptr_ra=[[LOOP_BEGIN_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_loop_[[SCHED]]_end: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}, codeptr_ra=[[LOOP_END_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[LOOP_END_RETURN_ADDRESS]]

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}, invoker={{[0-9]+}}, codeptr_ra=[[PARALLEL_RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[PARALLEL_RETURN_ADDRESS]]
  
  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_loop_[[SCHED]]_begin: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}, codeptr_ra=0x{{[0-f]+}}
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_loop_[[SCHED]]_end: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}, codeptr_ra=[[LOOP_END_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[THREAD_ID]]: fuzzy_address={{.*}}[[LOOP_END_RETURN_ADDRESS]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_loop_[[SCHED]]_begin: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}, codeptr_ra=0x{{[0-f]+}}
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_loop_[[SCHED]]_end: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}, codeptr_ra=[[LOOP_END_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[THREAD_ID]]: fuzzy_address={{.*}}[[LOOP_END_RETURN_ADDRESS]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_loop_[[SCHED]]_begin: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}, codeptr_ra=0x{{[0-f]+}}
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_loop_[[SCHED]]_end: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}, codeptr_ra=[[LOOP_END_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[THREAD_ID]]: fuzzy_address={{.*}}[[LOOP_END_RETURN_ADDRESS]]


  // CHECK-LOOP: 0: NULL_POINTER=[[NULL:.*$]]
  // CHECK-LOOP: 0: Schedule: [[SCHED:[a-z]+]]
  // CHECK-LOOP: 0: ompt_event_runtime_shutdown
  // CHECK-LOOP: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin: parent_task_id={{[0-9]+}}, parent_task_frame.exit=[[NULL]], parent_task_frame.reenter={{0x[0-f]+}}, parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=4, codeptr_ra={{0x[0-f]+}}, invoker={{[0-9]+}}
  // CHECK-LOOP: {{^}}[[MASTER_ID]]: ompt_event_loop_[[SCHED]]_begin: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}, codeptr_ra=[[LOOP_BEGIN_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK-LOOP: {{^}}{{[0-9]+}}: fuzzy_address={{.*}}[[LOOP_BEGIN_RETURN_ADDRESS]]
  // CHECK-LOOP: {{^}}{{[0-9]+}}: fuzzy_address={{.*}}[[LOOP_BEGIN_RETURN_ADDRESS]]
  // CHECK-LOOP: {{^}}{{[0-9]+}}: fuzzy_address={{.*}}[[LOOP_BEGIN_RETURN_ADDRESS]]
  // CHECK-LOOP: {{^}}{{[0-9]+}}: fuzzy_address={{.*}}[[LOOP_BEGIN_RETURN_ADDRESS]]
  // clang-format on

  return 0;
}
