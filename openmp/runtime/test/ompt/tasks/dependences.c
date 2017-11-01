// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt

#include "callback.h"
#include <omp.h>   
#include <math.h>
#include <unistd.h>

int main()
{
  int x = 0;
  #pragma omp parallel num_threads(2)
  {
    #pragma omp master
    {  
      #pragma omp task depend(out:x)
      {
        x++;
        delay(100);
      }
      print_fuzzy_address(1);
    
      #pragma omp task depend(in:x)
      {
        x = -1;
      }
    }
  }

  x++;


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_dependences'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_dependence'
  
  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]

  // make sure initial data pointers are null
  // CHECK-NOT: 0: new_task_data initially not null

  // CHECK: {{^}}{{[0-9]+}}: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter=[[NULL]], new_task_id=[[FIRST_TASK:[0-f]+]], codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}, task_type=ompt_task_explicit=4, has_dependences=yes
  // CHECK: {{^}}{{[0-9]+}}: ompt_event_task_dependences: task_id=[[FIRST_TASK]], deps={{0x[0-f]+}}, ndeps=1
  // CHECK: {{^}}{{[0-9]+}}: fuzzy_address={{.*}}[[RETURN_ADDRESS]]

  // CHECK: {{^}}{{[0-9]+}}: ompt_event_task_create: parent_task_id={{[0-9]+}}, parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter=[[NULL]], new_task_id=[[SECOND_TASK:[0-f]+]], codeptr_ra={{0x[0-f]+}}, task_type=ompt_task_explicit=4, has_dependences=yes
  // CHECK: {{^}}{{[0-9]+}}: ompt_event_task_dependences: task_id=[[SECOND_TASK]], deps={{0x[0-f]+}}, ndeps=1
  // CHECK: {{^}}{{[0-9]+}}: ompt_event_task_dependence_pair: first_task_id=[[FIRST_TASK]], second_task_id=[[SECOND_TASK]]


  return 0;
}
