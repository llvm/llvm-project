// RUN: %libomp-compile && env OMP_NUM_THREADS='3' \
// RUN:    %libomp-run | %sort-threads | FileCheck %s
// REQUIRES: ompt

// Checked gcc 10.1 still does not support detach clause on task construct.
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8, gcc-9, gcc-10
// gcc 11 introduced detach clause, but gomp interface in libomp has no support
// XFAIL: gcc-11, gcc-12
// clang supports detach clause since version 11.
// UNSUPPORTED: clang-10, clang-9, clang-8, clang-7
// icc compiler does not support detach clause.
// UNSUPPORTED: icc

#include "callback.h"
#include <omp.h>

int main() {
#pragma omp parallel
#pragma omp master
  {
    omp_event_handle_t event;
    omp_event_handle_t *f_event;
#pragma omp task detach(event) depend(out : f_event) shared(f_event) if (0)
    {
      printf("task 1\n");
      f_event = &event;
    }
#pragma omp task depend(in : f_event)
    { printf("task 2\n"); }
    printf("calling omp_fulfill_event\n");
    omp_fulfill_event(*f_event);
#pragma omp taskwait
  }
  return 0;
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

// CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin:
// CHECK-SAME: parent_task_id=[[PARENT_TASK_ID:[0-9]+]],
// CHECK-SAME: parent_task_frame.exit=[[NULL]],
// CHECK-SAME: parent_task_frame.reenter=0x{{[0-f]+}},
// CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-9]+]],
// CHECK-SAME: requested_team_size=3,

// CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin:
// CHECK-SAME: parallel_id=[[PARALLEL_ID]],
// CHECK-SAME: task_id=[[IMPLICIT_TASK_ID:[0-9]+]]

// The following is to match the taskwait task created in __kmpc_omp_wait_deps
// this should go away, once codegen for "detached if(0)" is fixed

// CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
// CHECK-SAME: parent_task_id=[[IMPLICIT_TASK_ID]],
// CHECK-SAME: has_dependences=yes

// CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
// CHECK-SAME: parent_task_id=[[IMPLICIT_TASK_ID]],
// CHECK-SAME: parent_task_frame.exit=0x{{[0-f]+}},
// CHECK-SAME: parent_task_frame.reenter=0x{{[0-f]+}},
// CHECK-SAME: new_task_id=[[TASK_ID:[0-9]+]],

// CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_task_schedule:
// CHECK-SAME: first_task_id=[[IMPLICIT_TASK_ID]],
// CHECK-SAME: second_task_id=[[TASK_ID]],
// CHECK-SAME: prior_task_status=ompt_task_switch=7

// CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_task_schedule:
// CHECK-SAME: first_task_id=[[TASK_ID]],
// CHECK-SAME: second_task_id=[[IMPLICIT_TASK_ID]],
// CHECK-SAME: prior_task_status=ompt_task_detach=4

// CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_task_schedule:
// CHECK-SAME: first_task_id=[[TASK_ID]],
// CHECK-SAME: second_task_id=18446744073709551615,
// CHECK-SAME: prior_task_status=ompt_task_late_fulfill=6
