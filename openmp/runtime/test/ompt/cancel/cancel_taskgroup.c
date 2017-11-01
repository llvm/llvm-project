// RUN:  %libomp-compile && env OMP_CANCELLATION=true %libomp-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// Current GOMP interface implementation does not support cancellation
// XFAIL: gcc

#include "callback.h"
#include <unistd.h>  
#include <stdio.h>

int main()
{
  int condition=0;
  #pragma omp parallel num_threads(2)
  {}

  print_frame(0);
  #pragma omp parallel num_threads(2)
  {
    #pragma omp master
    {
      #pragma omp taskgroup
      {
        #pragma omp task shared(condition)
        {
          printf("start execute task 1\n");
          OMPT_SIGNAL(condition);
          OMPT_WAIT(condition,2);
          #pragma omp cancellation point taskgroup
          printf("end execute task 1\n");
        }
        #pragma omp task shared(condition)
        {
          printf("start execute task 2\n");
          OMPT_SIGNAL(condition);
          OMPT_WAIT(condition,2);
          #pragma omp cancellation point taskgroup
          printf("end execute task 2\n");
        }
      #pragma omp task shared(condition)
        {
          printf("start execute task 3\n");
          OMPT_SIGNAL(condition);
          OMPT_WAIT(condition,2);
          #pragma omp cancellation point taskgroup
          printf("end execute task 3\n");
        }
      #pragma omp task if(0) shared(condition)
        {
          printf("start execute task 4\n");
          OMPT_WAIT(condition,1);
          #pragma omp cancel taskgroup
          printf("end execute task 4\n");
        }
        OMPT_SIGNAL(condition);
      }
    }
    #pragma omp barrier
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_master'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_schedule'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_cancel'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_thread_begin'


  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_master_begin: parallel_id=[[PARALLEL_ID:[0-9]+]], task_id=[[PARENT_TASK_ID:[0-9]+]], codeptr_ra={{0x[0-f]*}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK_ID]], parent_task_frame.exit={{0x[0-f]*}}, parent_task_frame.reenter={{0x[0-f]*}}, new_task_id=[[FIRST_TASK_ID:[0-9]+]], codeptr_ra={{0x[0-f]*}}, task_type=ompt_task_explicit=4, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK_ID]], parent_task_frame.exit={{0x[0-f]*}}, parent_task_frame.reenter={{0x[0-f]*}}, new_task_id=[[SECOND_TASK_ID:[0-9]+]], codeptr_ra={{0x[0-f]*}}, task_type=ompt_task_explicit=4, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK_ID]], parent_task_frame.exit={{0x[0-f]*}}, parent_task_frame.reenter={{0x[0-f]*}}, new_task_id=[[THIRD_TASK_ID:[0-9]+]], codeptr_ra={{0x[0-f]*}}, task_type=ompt_task_explicit=4, has_dependences=no

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK_ID]], parent_task_frame.exit={{0x[0-f]*}}, parent_task_frame.reenter={{0x[0-f]*}}, new_task_id=[[CANCEL_TASK_ID:[0-9]+]], codeptr_ra={{0x[0-f]*}}, task_type=ompt_task_explicit|ompt_task_undeferred=134217732, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_schedule: first_task_id=[[PARENT_TASK_ID]], second_task_id=[[CANCEL_TASK_ID]], prior_task_status=ompt_task_others=4
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_cancel: task_data=[[CANCEL_TASK_ID]], flags=ompt_cancel_taskgroup|ompt_cancel_activated=24, codeptr_ra={{0x[0-f]*}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_schedule: first_task_id=[[CANCEL_TASK_ID]], second_task_id=[[PARENT_TASK_ID]], prior_task_status=ompt_task_cancel=3

  // CHECK-DAG: {{^}}{{[0-9]+}}: ompt_event_cancel: task_data={{[0-9]+}}, flags=ompt_cancel_taskgroup|ompt_cancel_discarded_task=72, codeptr_ra=[[NULL]]
  // CHECK-DAG: {{^}}{{[0-9]+}}: ompt_event_cancel: task_data={{[0-9]+}}, flags=ompt_cancel_taskgroup|ompt_cancel_discarded_task=72, codeptr_ra=[[NULL]]
  
  // CHECK-DAG: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_thread_begin: thread_type=ompt_thread_worker=2, thread_id=[[THREAD_ID]]
  // CHECK-DAG: {{^}}[[THREAD_ID]]: ompt_event_cancel: task_data={{[0-9]+}}, flags=ompt_cancel_taskgroup|ompt_cancel_detected=40, codeptr_ra={{0x[0-f]*}}

  return 0;
}
