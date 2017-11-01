// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

int main()
{
  int x = 0;

  //implicit barrier at end of a parallel region
  #pragma omp parallel num_threads(2)
  {
    #pragma omp atomic
    x++;
  }
  print_fuzzy_address();


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region_wait'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // master thread implicit barrier at parallel end
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_barrier_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]


  // worker thread implicit barrier at parallel end
  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_barrier_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[NULL]]

  return 0;
}
