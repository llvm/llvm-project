// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// GCC generates code that does not call the runtime for the flush construct
// XFAIL: gcc

#include "callback.h"
#include <omp.h>

int main()
{
  #pragma omp parallel num_threads(2)
  {
    int tid = omp_get_thread_num();
    
    #pragma omp flush
    print_current_address(1);
  }

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_flush'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_flush: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: current_address=[[RETURN_ADDRESS]]
  //
  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_flush: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]
  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: current_address=[[RETURN_ADDRESS]]



  return 0;
}
