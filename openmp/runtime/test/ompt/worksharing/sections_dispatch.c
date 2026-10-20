// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: intel-15.0, intel-16.0, intel-17.0, intel-18.0
// GCC generates code that does not distinguish between sections and loops
// UNSUPPORTED: gcc
// clang-format on

#include "callback.h"
#include <omp.h>

int main() {
#pragma omp parallel sections num_threads(2)
  {
#pragma omp section
    {
      printf("%lu: section 1\n", ompt_get_thread_data()->value);
    }
#pragma omp section
    {
      printf("%lu: section 2\n", ompt_get_thread_data()->value);
    }
  }

  // clang-format off
  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_work'
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_dispatch'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_sections_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID:[0-f]+]],
  // CHECK-SAME: task_id=[[TASK_ID:[0-f]+]],
  // CHECK-SAME: codeptr_ra=[[SECT_BEGIN:(0x)?[0-f]+]], count=2
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_section_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]]
  // CHECK-SAME: codeptr_ra=[[SECT_BEGIN]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_sections_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-f]+}},
  // CHECK-SAME: codeptr_ra=[[SECT_END:(0x)?[0-f]+]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_sections_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID:[0-f]+]],
  // CHECK-SAME: codeptr_ra=[[SECT_BEGIN]], count=2
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_section_begin:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id=[[TASK_ID]]
  // CHECK-SAME: codeptr_ra=[[SECT_BEGIN]]
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_sections_end:
  // CHECK-SAME: parallel_id=[[PARALLEL_ID]], task_id={{[0-f]+}},
  // CHECK-SAME: codeptr_ra=[[SECT_END]]
  // clang-format on

  return 0;
}
