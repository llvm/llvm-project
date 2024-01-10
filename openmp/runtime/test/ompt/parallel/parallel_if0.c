// XFAIL: gcc, icc, clang-3, clang-4, clang-5, clang-6, clang-7, clang-8, clang-9, clang-10, clang-11, clang-12, clang-13, clang-14, clang-15, clang-16, clang-17
// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
#include "callback.h"

int main()
{
//  print_frame(0);
  #pragma omp parallel if(0)
  {
//    print_frame(1);
    print_ids(0);
    print_ids(1);
//    print_frame(0);
    #pragma omp parallel if(0)
    {
//      print_frame(1);
      print_ids(0);
      print_ids(1);
      print_ids(2);
//      print_frame(0);
      #pragma omp task
      {
//        print_frame(1);
        print_ids(0);
        print_ids(1);
        print_ids(2);
        print_ids(3);
      }
    }
    print_fuzzy_address(1);
  }
  print_fuzzy_address(2);

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_end'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_implicit_task_begin'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_implicit_task_end'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // make sure initial data pointers are null
  // CHECK-NOT: 0: parallel_data initially not null
  // CHECK-NOT: 0: task_data initially not null
  // CHECK-NOT: 0: thread_data initially not null

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter={{0x[0-f]+}}, parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=1, codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}, invoker=[[PARALLEL_INVOKER:[0-9]+]]

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 1: parallel_id=[[IMPLICIT_PARALLEL_ID:[0-9]+]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin: parent_task_id=[[IMPLICIT_TASK_ID]], parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, parallel_id=[[NESTED_PARALLEL_ID:[0-9]+]], requested_team_size=1, codeptr_ra=[[NESTED_RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}, invoker=[[PARALLEL_INVOKER]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 1: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: task level 2: parallel_id=[[IMPLICIT_PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], exit_frame=[[NULL]], reenter_frame={{0x[0-f]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create: parent_task_id=[[NESTED_IMPLICIT_TASK_ID]], parent_task_frame.exit={{0x[0-f]+}}, parent_task_frame.reenter={{0x[0-f]+}}, new_task_id=[[EXPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_schedule: first_task_id=[[NESTED_IMPLICIT_TASK_ID]], second_task_id=[[EXPLICIT_TASK_ID]], prior_task_status=ompt_task_switch=7
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[EXPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 1: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[NESTED_IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: task level 2: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}, reenter_frame={{0x[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_schedule: first_task_id=[[EXPLICIT_TASK_ID]], second_task_id=[[NESTED_IMPLICIT_TASK_ID]], prior_task_status=ompt_task_complete=1
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_end: task_id=[[EXPLICIT_TASK_ID]]


  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id=0, task_id=[[NESTED_IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end: parallel_id=[[NESTED_PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]
  // CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[NESTED_RETURN_ADDRESS]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end: parallel_id=[[PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], invoker=[[PARALLEL_INVOKER]]
  // CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]

  return 0;
}
