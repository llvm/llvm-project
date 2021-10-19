// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt

// GCC 9 introduced codegen for mutexinoutset
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8

// icc does not yet support mutexinoutset
// XFAIL: icc

// clang 9 introduced codegen for mutexinoutset
// UNSUPPORTED: clang-4, clang-5, clang-6, clang-7, clang-8

#include "callback.h"
#include <omp.h>
#include <math.h>
#include <unistd.h>

int main() {
  int x = 0;
#pragma omp parallel num_threads(2)
  {
#pragma omp master
    {
      print_ids(0);
      printf("%" PRIu64 ": address of x: %p\n", ompt_get_thread_data()->value,
             &x);
#pragma omp task depend(out : x)
      {
        x++;
        delay(100);
      }
      print_fuzzy_address(1);
      print_ids(0);

#pragma omp task depend(mutexinoutset : x)
      {
        x++;
        delay(100);
      }
      print_fuzzy_address(2);
      print_ids(0);

#pragma omp task depend(in : x)
      { x = -1; }
      print_ids(0);
    }
  }

  x++;

  return 0;
}

// Check if libomp supports the callbacks for this test.
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_create'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_dependences'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_task_depende

// CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]

// make sure initial data pointers are null
// CHECK-NOT: 0: new_task_data initially not null

// CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_implicit_task_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]],
// CHECK-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]

// CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]],
// CHECK-SAME: task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT:0x[0-f]+]],
// CHECK-SAME: reenter_frame=[[NULL]]

// CHECK: {{^}}[[MASTER_ID]]: address of x: [[ADDRX:0x[0-f]+]]
// CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
// CHECK-SAME: parent_task_id={{[0-9]+}}, parent_task_frame.exit=[[EXIT]],
// CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}},
// CHECK-SAME: new_task_id=[[FIRST_TASK:[0-f]+]],
// CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}},
// CHECK-SAME: task_type=ompt_task_explicit=4, has_dependences=yes

// CHECK: {{^}}[[MASTER_ID]]: ompt_event_dependences:
// CHECK-SAME: task_id=[[FIRST_TASK]], deps=[([[ADDRX]],
// CHECK-SAME: ompt_dependence_type_inout)], ndeps=1

// CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
// CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]],
// CHECK-SAME: task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]],
// CHECK-SAME: reenter_frame=[[NULL]]

// CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
// CHECK-SAME: parent_task_id={{[0-9]+}}, parent_task_frame.exit=[[EXIT]],
// CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}},
// CHECK-SAME: new_task_id=[[SECOND_TASK:[0-f]+]],
// CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS:0x[0-f]+]]{{[0-f][0-f]}},
// CHECK-SAME: task_type=ompt_task_explicit=4, has_dependences=yes

// CHECK: {{^}}[[MASTER_ID]]: ompt_event_dependences:
// CHECK-SAME: task_id=[[SECOND_TASK]], deps=[([[ADDRX]],
// CHECK-SAME: ompt_dependence_type_mutexinoutset)], ndeps=1

// CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS]]
// CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]],
// CHECK-SAME: task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]],
// CHECK-SAME: reenter_frame=[[NULL]]

// CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
// CHECK-SAME: parent_task_id={{[0-9]+}}, parent_task_frame.exit=[[EXIT]],
// CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}},
// CHECK-SAME: new_task_id=[[THIRD_TASK:[0-f]+]], codeptr_ra={{0x[0-f]+}},
// CHECK-SAME: task_type=ompt_task_explicit=4, has_dependences=yes

// CHECK: {{^}}[[MASTER_ID]]: ompt_event_dependences:
// CHECK-SAME: task_id=[[THIRD_TASK]], deps=[([[ADDRX]],
// CHECK-SAME: ompt_dependence_type_in)], ndeps=1

// CHECK: {{^}}[[MASTER_ID]]: task level 0: parallel_id=[[PARALLEL_ID]],
// CHECK-SAME: task_id=[[IMPLICIT_TASK_ID]], exit_frame=[[EXIT]],
// CHECK-SAME: reenter_frame=[[NULL]]
