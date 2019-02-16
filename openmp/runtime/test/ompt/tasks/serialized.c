// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
#define TEST_NEED_PRINT_FRAME_FROM_OUTLINED_FN
#include "callback.h"
#include <omp.h>
#include <math.h>

int main() {
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
      void *creator_frame = get_frame_address(0);
      int t = (int)sin(0.1);
#pragma omp task if (t)
      {
        void *task_frame = get_frame_address(0);
        if (creator_frame == task_frame) {
          // Assume this code was inlined which the compiler is allowed to do.
          print_frame(0);
        } else {
          // The exit frame must be our parent!
          print_frame_from_outlined_fn(1);
        }
        print_ids(0);
        print_ids(1);
        print_ids(2);
      }
      print_fuzzy_address(1);
      print_ids(0);
    }
    print_ids(0);
  }

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback

  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]

  // make sure initial data pointers are null
  // CHECK-NOT: 0: new_task_data initially not null

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_task_create
  // CHECK-SAME: parent_task_id={{[0-9]+}}, parent_task_frame.exit=[[NULL]]
  // CHECK-SAME: parent_task_frame.reenter=[[NULL]]
  // CHECK-SAME: new_task_id={{[0-9]+}}, codeptr_ra=[[NULL]]
  // CHECK-SAME: task_type=ompt_task_initial=1, has_dependences=no
  // CHECK: {{^}}[[MASTER_ID]]: __builtin_frame_address(0)
  // CHECK-SAME: =[[MAIN_REENTER:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_begin
  // CHECK-SAME: parent_task_id=[[PARENT_TASK_ID:[0-9]+]]
  // CHECK-SAME: parent_task_frame.exit=[[NULL]]
  // CHECK-SAME: parent_task_frame.reenter=0x{{[0-f]+}}
  // CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=2
  // CHECK-SAME: codeptr_ra=0x{{[0-f]+}}, invoker={{[0-9]+}}

  // nested parallel masters
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: __builtin_frame_address
  // CHECK-SAME: =[[EXIT:0x[0-f]+]]

  // CHECK: {{^}}[[MASTER_ID]]: task level 0
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK-SAME: exit_frame=[[EXIT]], reenter_frame=[[NULL]]

  // CHECK: {{^}}[[MASTER_ID]]: task level 1
  // CHECK-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID:[0-9]+]]
  // CHECK-SAME: task_id=[[PARENT_TASK_ID]],
  // CHECK-SAME: exit_frame=[[NULL]], reenter_frame=0x{{[0-f]+}}

  // CHECK: {{^}}[[MASTER_ID]]: __builtin_frame_address(0)=[[REENTER:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create
  // CHECK-SAME: parent_task_id=[[IMPLICIT_TASK_ID]]
  // CHECK-SAME: parent_task_frame.exit=[[EXIT]]
  // CHECK-SAME: parent_task_frame.reenter=0x{{[0-f]+}}
  // CHECK-SAME: new_task_id=[[TASK_ID:[0-9]+]]
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_schedule:
  // CHECK-SAME: first_task_id=[[IMPLICIT_TASK_ID]], second_task_id=[[TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: __builtin_frame_address
  // CHECK-SAME: =[[TASK_EXIT:0x[0-f]+]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]]
  // CHECK-SAME: exit_frame=[[TASK_EXIT]], reenter_frame=[[NULL]]

  // CHECK: {{^}}[[MASTER_ID]]: task level 1
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK-SAME: exit_frame=[[EXIT]], reenter_frame=0x{{[0-f]+}}

  // CHECK: {{^}}[[MASTER_ID]]: task level 2
  // CHECK-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]]
  // CHECK-SAME: task_id=[[PARENT_TASK_ID]]
  // CHECK-SAME: exit_frame=[[NULL]], reenter_frame=0x{{[0-f]+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_schedule
  // CHECK-SAME: first_task_id=[[TASK_ID]], second_task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_end: task_id=[[TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]

  // CHECK: {{^}}[[MASTER_ID]]: task level 0
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK-SAME: exit_frame=[[EXIT]], reenter_frame=[[NULL]]

  // implicit barrier parallel
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_begin
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: task level 0
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK-SAME: exit_frame=[[NULL]], reenter_frame=[[NULL]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_barrier_end
  // parallel_id is 0 because the region ended in the barrier!
  // CHECK-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end
  // CHECK-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_implicit_task_begin
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[THREAD_ID]]: __builtin_frame_address
  // CHECK-SAME: =[[EXIT:0x[0-f]+]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 0
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK-SAME: exit_frame=[[EXIT]], reenter_frame=[[NULL]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 1
  // CHECK-SAME: parallel_id=[[IMPLICIT_PARALLEL_ID]]
  // CHECK-SAME: task_id=[[PARENT_TASK_ID]]
  // CHECK-SAME: exit_frame=[[NULL]], reenter_frame=0x{{[0-f]+}}

  // CHECK: {{^}}[[THREAD_ID]]: __builtin_frame_address(0)={{0x[0-f]+}}
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_begin
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[THREAD_ID]]: task level 0
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK-SAME: exit_frame=[[NULL]], reenter_frame=[[NULL]]
  // parallel_id is 0 because the region ended in the barrier!
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_barrier_end
  // CHECK-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]

  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_implicit_task_end
  // CHECK-SAME: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]

  return 0;
}
