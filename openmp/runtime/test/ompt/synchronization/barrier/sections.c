// RUN: %libomp-compile-and-run | %sort-threads  | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>

int main()
{
  int x = 0;

  #pragma omp parallel num_threads(2)
  {
    //implicit barrier after sections with nowait but with lastprivates
    //implicit barrier at end of sections
    #pragma omp sections
    {
      #pragma omp section 
      {
        #pragma omp atomic
        x++;
      }
      
      #pragma omp section 
      {
        #pragma omp atomic
        x++;
      }
    }
  }

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_sync_region_wait'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // master thread implicit barrier at sections end
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_barrier_implicit_workshare_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_workshare_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_workshare_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_workshare_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}

  // master thread implicit barrier at parallel end
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_parallel_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_wait_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}


  // worker thread implicit barrier at sections end
  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_barrier_implicit_workshare_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_implicit_workshare_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_implicit_workshare_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_workshare_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra={{0x[0-f]+}}

  // worker thread implicit barrier at parallel end
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_implicit_parallel_begin: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_wait_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_implicit_parallel_end: parallel_id={{[0-9]+}}, task_id={{[0-9]+}}, codeptr_ra=[[NULL]]
  // clang-format on

  return 0;
}
