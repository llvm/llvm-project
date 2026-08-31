// clang-format off
// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7
// clang-format on
#include "callback.h"

int main(void) {
#pragma omp parallel num_threads(1)
#pragma omp sections
  {
#pragma omp section
    {
    }
  }
}

// clang-format off
// Check if libomp supports the callbacks for this test.
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_begin'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_parallel_end'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_implicit_task_begin'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_implicit_task_end'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_work'
// CHECK-NOT: {{^}}0: Could not register callback 'ompt_event_dispatch'

// CHECK: 0: NULL_POINTER=[[NULL:.*$]]

// CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin: parent_task_id=[[PARENT_TASK_ID:[0-f]+]], parent_task_frame.exit=[[NULL]], parent_task_frame.reenter={{(0x)?[0-f]+}}, parallel_id=[[PARALLEL_ID:[0-f]+]], requested_team_size=1, codeptr_ra=[[RETURN_ADDRESS:(0x)?[0-f]+]]{{[0-f][0-f]}}, invoker=[[PARALLEL_INVOKER:[0-9]+]]
// CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-f]+]]
// CHECK: {{^}}[[MASTER_ID]]: ompt_event_sections_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
// CHECK: {{^}}[[MASTER_ID]]: ompt_event_section_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
// CHECK: {{^}}[[MASTER_ID]]: ompt_event_sections_end: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
// CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id=0, task_id=[[IMPLICIT_TASK_ID]]
// CHECK: {{^}}[[MASTER_ID]]: ompt_event_parallel_end: parallel_id=[[PARALLEL_ID]], task_id=[[PARENT_TASK_ID]], invoker=[[PARALLEL_INVOKER]], codeptr_ra=[[RETURN_ADDRESS]]{{[0-f][0-f]}}

// clang-format on
