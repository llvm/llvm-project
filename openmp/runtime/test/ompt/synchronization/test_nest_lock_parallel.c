// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
// clang-format on
#include "callback.h"
#include <omp.h>

int main() {
  omp_nest_lock_t nest_lock;
  omp_init_nest_lock(&nest_lock);

#pragma omp parallel num_threads(2)
  {
#pragma omp master
    {
      omp_set_nest_lock(&nest_lock);
      print_fuzzy_address(1);
    }
#pragma omp barrier
    omp_test_nest_lock(&nest_lock); // should fail for non-master
    print_fuzzy_address(2);
#pragma omp barrier
#pragma omp master
    {
      omp_unset_nest_lock(&nest_lock);
      print_fuzzy_address(3);
      omp_unset_nest_lock(&nest_lock);
      print_fuzzy_address(4);
    }
  }

  omp_destroy_nest_lock(&nest_lock);

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquire'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquired'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_released'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_nest_lock'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_wait_nest_lock: wait_id=[[WAIT_ID:[0-9]+]], hint=0, impl={{[0-9]+}}, codeptr_ra=[[RETURN_ADDRESS:(0x)?[0-f]+]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_nest_lock_first: wait_id=[[WAIT_ID]], codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_test_nest_lock: wait_id=[[WAIT_ID]], hint=0, impl={{[0-9]+}}, codeptr_ra=[[RETURN_ADDRESS:(0x)?[0-f]+]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_nest_lock_next: wait_id=[[WAIT_ID]], codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_nest_lock_prev: wait_id=[[WAIT_ID]], codeptr_ra=[[RETURN_ADDRESS:(0x)?[0-f]+]]{{[0-f][0-f]}}
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_nest_lock_last: wait_id=[[WAIT_ID]], codeptr_ra=[[RETURN_ADDRESS:(0x)?[0-f]+]]{{[0-f][0-f]}}
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_destroy_nest_lock: wait_id=[[WAIT_ID]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_wait_test_nest_lock: wait_id=[[WAIT_ID]], hint=0, impl={{[0-9]+}}, codeptr_ra=[[RETURN_ADDRESS:(0x)?[0-f]+]]{{[0-f][0-f]}}
  // CHECK-NOT: {{^}}[[THREAD_ID]]: ompt_event_acquired_test_nest_lock_next: wait_id=[[WAIT_ID]]
  // CHECK-NEXT: {{^}}[[THREAD_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
  // clang-format on

  return 0;
}
