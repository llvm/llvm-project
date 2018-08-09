// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
#include "callback.h"
#include <omp.h>
#include <math.h>

int main() {
  //initialize the OpenMP runtime
  omp_get_num_threads();

  // initial task
  print_ids(0);

  int x;
// implicit task
#pragma omp parallel num_threads(1)
  {
    print_ids(0);
    x++;
  }

#pragma omp parallel num_threads(2)
  {
// explicit task
#pragma omp single
#pragma omp task
    {
      print_ids(0);
      x++;
    }
// explicit task with undeferred
#pragma omp single
#pragma omp task if (0)
    {
      print_ids(0);
      x++;
    }

// explicit task with untied
#pragma omp single
#pragma omp task untied
    {
      // Output of thread_id is needed to know on which thread task is executed
      printf("%" PRIu64 ": explicit_untied\n", ompt_get_thread_data()->value);
      print_ids(0);
      print_frame(1);
      x++;
#pragma omp taskyield
      printf("%" PRIu64 ": explicit_untied(2)\n",
             ompt_get_thread_data()->value);
      print_ids(0);
      print_frame(1);
      x++;
#pragma omp taskwait
      printf("%" PRIu64 ": explicit_untied(3)\n",
             ompt_get_thread_data()->value);
      print_ids(0);
      print_frame(1);
      x++;
    }
// explicit task with final
#pragma omp single
#pragma omp task final(1)
    {
      print_ids(0);
      x++;
// nested explicit task with final and undeferred
#pragma omp task
      {
        print_ids(0);
        x++;
      }
    }

    // Mergeable task test deactivated for now
    // explicit task with mergeable
    /*
    #pragma omp task mergeable if((int)sin(0))
    {
      print_ids(0);
      x++;
    }
    */

    // TODO: merged task
  }

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'

  // CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_task_create: parent_task_id=0
  // CHECK-SAME: parent_task_frame.exit=[[NULL]]
  // CHECK-SAME: parent_task_frame.reenter=[[NULL]]
  // CHECK-SAME: new_task_id=[[INITIAL_TASK_ID:[0-9]+]], codeptr_ra=[[NULL]]
  // CHECK-SAME: task_type=ompt_task_initial=1, has_dependences=no

  // CHECK-NOT: 0: parallel_data initially not null

  // initial task
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id={{[0-9]+}}
  // CHECK-SAME: task_id=[[INITIAL_TASK_ID]], exit_frame=[[NULL]]
  // CHECK-SAME: reenter_frame=[[NULL]]
  // CHECK-SAME: task_type=ompt_task_initial=1, thread_num=0

  // implicit task
  // CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id={{[0-9]+}}
  // CHECK-SAME: task_id={{[0-9]+}}, exit_frame={{0x[0-f]+}}
  // CHECK-SAME: reenter_frame=[[NULL]]
  // CHECK-SAME: task_type=ompt_task_implicit|ompt_task_undeferred=134217730
  // CHECK-SAME: thread_num=0

  // explicit task
  // CHECK: {{^[0-9]+}}: ompt_event_task_create: parent_task_id={{[0-9]+}}
  // CHECK-SAME: parent_task_frame.exit={{0x[0-f]+}}
  // CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}}
  // CHECK-SAME: new_task_id=[[EXPLICIT_TASK_ID:[0-9]+]]
  // CHECK-SAME: codeptr_ra={{0x[0-f]+}}
  // CHECK-SAME: task_type=ompt_task_explicit=4
  // CHECK-SAME: has_dependences=no

  // CHECK: [[THREAD_ID_1:[0-9]+]]: ompt_event_task_schedule:
  // CHECK-SAME: second_task_id=[[EXPLICIT_TASK_ID]]

  // CHECK: [[THREAD_ID_1]]: task level 0: parallel_id=[[PARALLEL_ID:[0-9]+]]
  // CHECK-SAME: task_id=[[EXPLICIT_TASK_ID]], exit_frame={{0x[0-f]+}}
  // CHECK-SAME: reenter_frame=[[NULL]], task_type=ompt_task_explicit=4
  // CHECK-SAME: thread_num={{[01]}}

  // explicit task with undeferred
  // CHECK: {{^[0-9]+}}: ompt_event_task_create: parent_task_id={{[0-9]+}}
  // CHECK-SAME: parent_task_frame.exit={{0x[0-f]+}}
  // CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}}
  // CHECK-SAME: new_task_id=[[EXPLICIT_UNDEFERRED_TASK_ID:[0-9]+]]
  // CHECK-SAME: codeptr_ra={{0x[0-f]+}}
  // CHECK-SAME: task_type=ompt_task_explicit|ompt_task_undeferred=134217732
  // CHECK-SAME: has_dependences=no

  // CHECK: [[THREAD_ID_2:[0-9]+]]: ompt_event_task_schedule:
  // CHECK-SAME: second_task_id=[[EXPLICIT_UNDEFERRED_TASK_ID]]

  // CHECK: [[THREAD_ID_2]]: task level 0: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: task_id=[[EXPLICIT_UNDEFERRED_TASK_ID]]
  // CHECK-SAME: exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // CHECK-SAME: task_type=ompt_task_explicit|ompt_task_undeferred=134217732
  // CHECK-SAME: thread_num={{[01]}}

  // explicit task with untied
  // CHECK: {{^[0-9]+}}: ompt_event_task_create: parent_task_id={{[0-9]+}}
  // CHECK-SAME: parent_task_frame.exit={{0x[0-f]+}}
  // CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}}
  // CHECK-SAME: new_task_id=[[EXPLICIT_UNTIED_TASK_ID:[0-9]+]]
  // CHECK-SAME: codeptr_ra={{0x[0-f]+}}
  // CHECK-SAME: task_type=ompt_task_explicit|ompt_task_untied=268435460
  // CHECK-SAME: has_dependences=no

  // Here the thread_id cannot be taken from a schedule event as there
  // may be multiple of those
  // CHECK: [[THREAD_ID_3:[0-9]+]]: explicit_untied
  // CHECK: [[THREAD_ID_3]]: task level 0: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: task_id=[[EXPLICIT_UNTIED_TASK_ID]]
  // CHECK-SAME: exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // CHECK-SAME: task_type=ompt_task_explicit|ompt_task_untied=268435460
  // CHECK-SAME: thread_num={{[01]}}

  // after taskyield
  // CHECK: [[THREAD_ID_3_2:[0-9]+]]: explicit_untied(2)
  // CHECK: [[THREAD_ID_3_2]]: task level 0: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: task_id=[[EXPLICIT_UNTIED_TASK_ID]]
  // CHECK-SAME: exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // CHECK-SAME: task_type=ompt_task_explicit|ompt_task_untied=268435460
  // CHECK-SAME: thread_num={{[01]}}

  // after taskwait
  // CHECK: [[THREAD_ID_3_3:[0-9]+]]: explicit_untied(3)
  // CHECK: [[THREAD_ID_3_3]]: task level 0: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: task_id=[[EXPLICIT_UNTIED_TASK_ID]]
  // CHECK-SAME: exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // CHECK-SAME: task_type=ompt_task_explicit|ompt_task_untied=268435460
  // CHECK-SAME: thread_num={{[01]}}

  // explicit task with final
  // CHECK: {{^[0-9]+}}: ompt_event_task_create: parent_task_id={{[0-9]+}}
  // CHECK-SAME: parent_task_frame.exit={{0x[0-f]+}}
  // CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}}
  // CHECK-SAME: new_task_id=[[EXPLICIT_FINAL_TASK_ID:[0-9]+]]
  // CHECK-SAME: codeptr_ra={{0x[0-f]+}}
  // CHECK-SAME: task_type=ompt_task_explicit|ompt_task_final=536870916
  // CHECK-SAME: has_dependences=no

  // CHECK: [[THREAD_ID_4:[0-9]+]]: ompt_event_task_schedule:
  // CHECK-SAME: second_task_id=[[EXPLICIT_FINAL_TASK_ID]]

  // CHECK: [[THREAD_ID_4]]: task level 0: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: task_id=[[EXPLICIT_FINAL_TASK_ID]]
  // CHECK-SAME: exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // CHECK-SAME: task_type=ompt_task_explicit|ompt_task_final=536870916
  // CHECK-SAME: thread_num={{[01]}}

  // nested explicit task with final and undeferred
  // CHECK: {{^[0-9]+}}: ompt_event_task_create: parent_task_id={{[0-9]+}}
  // CHECK-SAME: parent_task_frame.exit={{0x[0-f]+}}
  // CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}}
  // CHECK-SAME: new_task_id=[[NESTED_FINAL_UNDEFERRED_TASK_ID:[0-9]+]]
  // CHECK-SAME: codeptr_ra={{0x[0-f]+}}
  // CHECK-SAME: task_type=ompt_task_explicit|ompt_task_undeferred
  // CHECK-SAME:|ompt_task_final=671088644
  // CHECK-SAME: has_dependences=no

  // CHECK: [[THREAD_ID_5:[0-9]+]]: ompt_event_task_schedule:
  // CHECK-SAME: second_task_id=[[NESTED_FINAL_UNDEFERRED_TASK_ID]]

  // CHECK: [[THREAD_ID_5]]: task level 0: parallel_id=[[PARALLEL_ID]]
  // CHECK-SAME: task_id=[[NESTED_FINAL_UNDEFERRED_TASK_ID]]
  // CHECK-SAME: exit_frame={{0x[0-f]+}}, reenter_frame=[[NULL]]
  // CHECK-SAME: task_type=ompt_task_explicit|ompt_task_undeferred
  // CHECK-SAME:|ompt_task_final=671088644
  // CHECK-SAME: thread_num={{[01]}}

  return 0;
}
