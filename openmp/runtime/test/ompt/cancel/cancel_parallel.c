// RUN: %libomp-compile && env OMP_CANCELLATION=true %libomp-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// Current GOMP interface implementation does not support cancellation
// XFAIL: gcc

#include "callback.h"
#include "omp.h"

int main()
{
  #pragma omp parallel num_threads(2)
  {
    if(omp_get_thread_num() == 0)
    {
      printf("%" PRIu64 ": fuzzy_address=0x%lx or 0x%lx\n", ompt_get_thread_data()->value, ((uint64_t)(char*)(&& ompt_label_1))/256-1, ((uint64_t)(char*)(&& ompt_label_1))/256);
      #pragma omp cancel parallel
      print_fuzzy_address(1); //does not actually print the address but provides a label
    }
    else
    {
      delay(100);
      printf("%" PRIu64 ": fuzzy_address=0x%lx or 0x%lx\n", ompt_get_thread_data()->value, ((uint64_t)(char*)(&& ompt_label_2))/256-1, ((uint64_t)(char*)(&& ompt_label_2))/256);
      #pragma omp cancellation point parallel
      print_fuzzy_address(2); //does not actually print the address but provides a label
    }
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_cancel'

  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter=[[NULL]], new_task_id=[[TASK_ID:[0-9]+]], codeptr_ra=[[NULL]], task_type=ompt_task_initial=1, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_cancel: task_data=[[TASK_ID:[0-9]+]], flags=ompt_cancel_parallel|ompt_cancel_activated=17, codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: fuzzy_address={{.*}}[[OTHER_RETURN_ADDRESS:0x[0-f]+]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_cancel: task_data=[[TASK_ID:[0-9]+]], flags=ompt_cancel_parallel|ompt_cancel_detected=33, codeptr_ra=[[OTHER_RETURN_ADDRESS]]{{[0-f][0-f]}}

  return 0;
}
