// RUN: %libomp-compile && env OMP_CANCELLATION=true %libomp-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// Current GOMP interface implementation does not support cancellation; icc 16 does not distinguish between sections and loops
// XFAIL: icc-16

#include "callback.h"
#include <unistd.h>

int main()
{
  int condition=0;
  #pragma omp parallel num_threads(2)
  {
    int x = 0;
    int i;
    #pragma omp for
    for(i = 0; i < 2; i++)
    {
      if(i == 0)
      {
        x++;
        OMPT_SIGNAL(condition);
        #pragma omp cancel for
      }
      else
      {
        x++;
        OMPT_WAIT(condition,1);
        delay(10000);
        #pragma omp cancellation point for
      }
    }
  }
  #pragma omp parallel num_threads(2)
  {
    #pragma omp sections
    {
      #pragma omp section
      {
        OMPT_SIGNAL(condition);
        #pragma omp cancel sections
      }
      #pragma omp section
      {
        OMPT_WAIT(condition,2);
        delay(10000);
        #pragma omp cancellation point sections
      }
    }
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_cancel'

  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_initial_task_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, actual_parallelism=1, index=1, flags=1
 
  // cancel for and sections
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_cancel: task_data=[[TASK_ID:[0-9]+]], flags=ompt_cancel_loop|ompt_cancel_activated=20, codeptr_ra={{0x[0-f]*}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_cancel: task_data=[[TASK_ID:[0-9]+]], flags=ompt_cancel_sections|ompt_cancel_{{activated=18|detected=34}}, codeptr_ra={{0x[0-f]*}}
  // CHECK: {{^}}[[OTHER_THREAD_ID:[0-9]+]]: ompt_event_cancel: task_data=[[TASK_ID:[0-9]+]], flags=ompt_cancel_loop|ompt_cancel_detected=36, codeptr_ra={{0x[0-f]*}}
  // CHECK: {{^}}[[OTHER_THREAD_ID:[0-9]+]]: ompt_event_cancel: task_data=[[TASK_ID:[0-9]+]], flags=ompt_cancel_sections|ompt_cancel_{{activated=18|detected=34}}, codeptr_ra={{0x[0-f]*}}

  return 0;
}
