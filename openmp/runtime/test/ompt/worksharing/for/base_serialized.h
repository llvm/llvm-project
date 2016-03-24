#include "callback.h"
#include <omp.h>

int main()
{
  unsigned int i;

  #pragma omp parallel for num_threads(1) schedule(SCHEDULE)
  for (i = 0; i < 1; i++) {
  }

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_parallel_begin: parent_task_id=[[PARENT_TASK_ID:[0-9]+]], parent_task_frame=0x{{[0-f]+}}, parallel_id=[[PARALLEL_ID:[0-9]+]], requested_team_size=1, parallel_function=0x{{[0-f]+}}, invoker={{.+}}

  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_begin: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID:[0-9]+]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_loop_begin: parallel_id=[[PARALLEL_ID]], parent_task_id=[[IMPLICIT_TASK_ID]], workshare_function=0x{{[0-f]+}}
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_loop_end: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_implicit_task_end: parallel_id=[[PARALLEL_ID]], task_id=[[IMPLICIT_TASK_ID]]

  return 0;
}
