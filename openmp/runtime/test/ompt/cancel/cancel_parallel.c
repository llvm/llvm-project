// RUN: %libomp-compile && env OMP_CANCELLATION=true %libomp-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// Current GOMP interface implementation does not support cancellation
// XFAIL: gcc

#include "callback.h"
#include "omp.h"

int main() {
  #pragma omp parallel num_threads(2)
  {
    if (omp_get_thread_num() == 0) {
      print_fuzzy_address_blocks(get_ompt_label_address(1));
      #pragma omp cancel parallel
      define_ompt_label(1);
      // We cannot print at this location because the parallel region is cancelled!
    } else {
      delay(100);
      print_fuzzy_address_blocks(get_ompt_label_address(2));
      #pragma omp cancellation point parallel
      define_ompt_label(2);
      // We cannot print at this location because the parallel region is cancelled!
    }
  }

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_cancel'

  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_task_create: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter=[[NULL]], new_task_id=[[TASK_ID:[0-9]+]], codeptr_ra=[[NULL]], task_type=ompt_task_initial=1, has_dependences=no
  // CHECK-DAG: {{^}}[[MASTER_ID]]: ompt_event_cancel: task_data=[[TASK_ID:[0-9]+]], flags=ompt_cancel_parallel|ompt_cancel_activated=17, codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK-DAG: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin
  // CHECK-DAG: {{^}}[[THREAD_ID]]: ompt_event_cancel: task_data=[[TASK_ID:[0-9]+]], flags=ompt_cancel_parallel|ompt_cancel_detected=33, codeptr_ra=[[OTHER_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK-DAG: {{^}}[[THREAD_ID]]: fuzzy_address={{.*}}[[OTHER_RETURN_ADDRESS]]

  return 0;
}
