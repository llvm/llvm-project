// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt
// GCC generates code that does not distinguish between sections and loops
// XFAIL: gcc

#include "callback.h"
#include <omp.h>

int main()
{
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

  // Check if libomp supports the callbacks for this test.
  // CHECK-NOT: {{^}}0: Could not register callback 'ompt_callback_work'

  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_sections_begin: parallel_id=[[PARALLEL_ID:[0-9]+]], parent_task_id=[[TASK_ID:[0-9]+]], codeptr_ra=[[SECT_BEGIN:0x[0-f]+]], count=2
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_sections_end: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}, codeptr_ra=[[SECT_END:0x[0-f]+]]

  // CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_sections_begin: parallel_id=[[PARALLEL_ID]], parent_task_id=[[TASK_ID:[0-9]+]], codeptr_ra=[[SECT_BEGIN]], count=2
  // CHECK: {{^}}[[THREAD_ID]]: ompt_event_sections_end: parallel_id=[[PARALLEL_ID]], task_id={{[0-9]+}}, codeptr_ra=[[SECT_END]]

  return 0;
}
