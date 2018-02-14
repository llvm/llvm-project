// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
#define TEST_NEED_PRINT_FRAME_FROM_OUTLINED_FN
#include "callback.h"
#include <omp.h> 

int main()
{
  int condition=0;
  omp_set_nested(0);
  print_frame(0);
  #pragma omp parallel num_threads(2)
  {
    print_frame_from_outlined_fn(1);
    print_ids(0);
    print_ids(1);
    print_frame(0);
    #pragma omp master
    {
      print_ids(0);
      #pragma omp task shared(condition)
      {
        OMPT_SIGNAL(condition);
        print_frame(1);
        print_ids(0);
        print_ids(1);
        print_ids(2);
      }
      print_fuzzy_address(1);
      OMPT_WAIT(condition,1);
      print_ids(0);
    }
    #pragma omp barrier
    print_ids(0);
  }


  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_schedule'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_end'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_implicit_task'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquire'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_acquired'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_mutex_released'


  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]

  // make sure initial data pointers are null
  // CHECK-NOT: 0: new_task_data initially not null
  
  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: __builtin_frame_address(0)=[[MAIN_REENTER:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter=[[MAIN_REENTER]], parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=2, codeptr_ra=0x{{[0-f]+}}, invoker=[[PARALLEL_INVOKER:[0-9]+]]
  // nested parallel masters
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: __builtin_frame_address({{.}})=[[EXIT:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[NULL]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 1: parallel_id=[[IMPLICIT_PARALLEL_ID:[0-9]+]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[MAIN_REENTER]]
  // CHECK: {{^}}[[MASTER_ID]]: __builtin_frame_address(0)=[[REENTER:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[NULL]]
  // <- ompt_event_task_create would be expected here
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id=[[IMPLICIT_TASK_ID]], parent_task_frame.exit=[[EXIT]], parent_task_frame.reenter=[[REENTER]], new_task_id=[[TASK_ID:[0-9]+]], codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}
  // CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[NULL]]
  // explicit barrier after master
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[REENTER]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_end: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // implicit barrier parallel
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[NULL]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[THREAD_ID]]: __builtin_frame_address({{.}})=[[EXIT:0x[0-f]+]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[MAIN_REENTER]]
  // CHECK: {{^}}[[THREAD_ID]]: __builtin_frame_address(0)=[[REENTER:0x[0-f]+]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[REENTER]]
  // this is expected to come earlier and at MASTER:
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_task_schedule: first_task_id=[[IMPLICIT_TASK_ID]], second_task_id=[[TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: __builtin_frame_address(1)=[[TASK_EXIT:0x[0-f]+]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]], exit_frame=[[TASK_EXIT]], reenter_frame=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 1: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]], reenter_frame=[[REENTER]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 2: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[MAIN_REENTER]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_task_schedule: first_task_id=[[TASK_ID]], second_task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_task_end: task_id=[[TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_end: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[NULL]], reenter_frame=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end: parallel_id={{[0-9]+}}, task_id=[[IMPLICIT_TASK_ID]]



  return 0;
}
