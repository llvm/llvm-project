// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// Current GOMP interface implements taskyield as stub
// XFAIL: gcc

#include "callback.h"
#include <omp.h>   
#include <unistd.h>

int main()
{
  int condition=0, x=0;
  #pragma omp parallel num_threads(2)
  {
    #pragma omp master
    {
        #pragma omp task shared(condition)
        {
          OMPT_SIGNAL(condition);
          OMPT_WAIT(condition,2);
        }
        OMPT_WAIT(condition,1);
        #pragma omp task shared(x)
        {
          x++;
        }
        printf("%" PRIu64 ": before yield\n", ompt_get_thread_data()->value);
        #pragma omp taskyield
        printf("%" PRIu64 ": after yield\n", ompt_get_thread_data()->value);
        OMPT_SIGNAL(condition);
    }
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_schedule'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'


  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]

  // make sure initial data pointers are null
  // CHECK-NOT: 0: new_task_data initially not null
  
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID:[0-9]+]], team_size={{[0-9]+}}, thread_num={{[0-9]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id=[[WORKER_TASK:[0-9]+]], codeptr_ra={{0x[0-f]+}}, task_type=ompt_task_explicit=4, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id=[[MAIN_TASK:[0-9]+]], codeptr_ra={{0x[0-f]+}}, task_type=ompt_task_explicit=4, has_dependences=no

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_schedule: first_task_id=[[IMPLICIT_TASK_ID]], second_task_id=[[MAIN_TASK]], prior_task_status=ompt_task_yield=2
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_schedule: first_task_id=[[MAIN_TASK]], second_task_id=[[IMPLICIT_TASK_ID]], prior_task_status=ompt_task_complete=1

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_task_schedule: first_task_id={{[0-9]+}}, second_task_id=[[WORKER_TASK]], prior_task_status=ompt_task_others=4
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_task_schedule: first_task_id=[[WORKER_TASK]], second_task_id={{[0-9]+}}, prior_task_status=ompt_task_complete=1





  return 0;
}
