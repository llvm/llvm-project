// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt

#include "callback.h"
#include <omp.h>

int main()
{
  omp_lock_t lock;
  omp_init_lock(&lock);
  print_fuzzy_address(1);

  omp_test_lock(&lock);
  print_fuzzy_address(2);
  omp_unset_lock(&lock);
  print_fuzzy_address(3);

  omp_set_lock(&lock);
  print_fuzzy_address(4);
  omp_test_lock(&lock);
  print_fuzzy_address(5);
  omp_unset_lock(&lock);
  print_fuzzy_address(6);

  omp_destroy_lock(&lock);
  print_fuzzy_address(7);

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquire'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquired'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_released'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_nest_lock'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_init_lock: wait_id=[[WAIT_ID:[0-9]+]], hint=0, impl={{[0-9]+}}, codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_test_lock: wait_id=[[WAIT_ID]], hint=0, impl={{[0-9]+}}, codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_test_lock: wait_id=[[WAIT_ID]], codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_lock: wait_id=[[WAIT_ID]], codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_lock: wait_id=[[WAIT_ID]], hint=0, impl={{[0-9]+}}, codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_acquired_lock: wait_id=[[WAIT_ID]], codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_test_lock: wait_id=[[WAIT_ID]], hint=0, impl={{[0-9]+}}, codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_release_lock: wait_id=[[WAIT_ID]], codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_destroy_lock: wait_id=[[WAIT_ID]], codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK-NEXT: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]

  return 0;
}
