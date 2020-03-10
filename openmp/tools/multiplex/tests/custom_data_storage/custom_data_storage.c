// RUN: %libomp-tool -DFIRST_TOOL -o %t.first.tool.so %s && \
// RUN: %libomp-tool -DSECOND_TOOL -o %t.second.tool.so %s && \
// RUN: %libomp-compile && \
// RUN: env OMP_TOOL_LIBRARIES=%t.first.tool.so \
// RUN: CUSTOM_DATA_STORAGE_TOOL_LIBRARIES=%t.second.tool.so \
// RUN: %libomp-run | %sort-threads | FileCheck %s

#if defined(FIRST_TOOL)
#include "first-tool.h"
#elif defined(SECOND_TOOL)
#include "second-tool.h"
#else /* APP */

#include "../ompt-signal.h"
#include "omp.h"
#include <stdio.h>

int main() {
  int x, s = 0;
#pragma omp parallel num_threads(2) shared(s)
  {
#pragma omp master
    {
#pragma omp task shared(s)
      {
        omp_control_tool(5, 1, NULL);
        OMPT_SIGNAL(s);
      }
    }
    if (omp_get_thread_num() == 1)
      OMPT_WAIT(s, 1);
  }
  return 0;
}
// Check if libomp supports the callbacks for this test.
// CHECK-NOT: {{^}}0: Could not register callback

// CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]
// CHECK: {{^}}0: NULL_POINTER=[[NULL]]
// CHECK: {{^}}0: ompt_event_runtime_shutdown
// CHECK: {{^}}0: ompt_event_runtime_shutdown

// CHECK: {{^}}[[_1ST_MSTR_TID:[0-9]+]]: _first_tool: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_initial=1,
// CHECK-SAME: thread_id=[[_1ST_MSTR_TID]]

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_initial_task_begin:
// CHECK-SAME: parallel_id=[[_FIRST_INIT_PARALLEL_ID:[0-9]+]],
// CHECK-SAME: task_id=[[_FIRST_INITIAL_TASK_ID:[0-9]+]], actual_parallelism=1,

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_parallel_begin:
// CHECK-SAME: parent_task_id=[[_FIRST_INITIAL_TASK_ID]],
// CHECK-SAME: parent_task_frame.exit=(nil),
// CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}},
// CHECK-SAME: parallel_id=[[_FIRST_PARALLEL_ID:[0-9]+]], requested_team_size=2,
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}, invoker=2

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_implicit_task_begin:
// CHECK-SAME: parallel_id=[[_FIRST_PARALLEL_ID]],
// CHECK-SAME: task_id=[[_FIRST_MASTER_IMPLICIT_TASK_ID:[0-9]+]], team_size=2,
// CHECK-SAME: thread_num=0

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_master_begin:
// CHECK-SAME: parallel_id=[[_FIRST_PARALLEL_ID]],
// CHECK-SAME: task_id=[[_FIRST_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_task_create:
// CHECK-SAME: parent_task_id=[[_FIRST_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: parent_task_frame.exit={{0x[0-f]+}},
// CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}},
// CHECK-SAME: new_task_id=[[_FIRST_EXPLICIT_TASK_ID:[0-9]+]],
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}, task_type=ompt_task_explicit=4,
// CHECK-SAME: has_dependences=no

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_master_end:
// CHECK-SAME: parallel_id=[[_FIRST_PARALLEL_ID]],
// CHECK-SAME: task_id=[[_FIRST_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_barrier_begin:
// CHECK-SAME: parallel_id=[[_FIRST_PARALLEL_ID]],
// CHECK-SAME: task_id=[[_FIRST_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_wait_barrier_begin:
// CHECK-SAME: parallel_id=[[_FIRST_PARALLEL_ID]],
// CHECK-SAME: task_id=[[_FIRST_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_task_schedule:
// CHECK-SAME: first_task_id=[[_FIRST_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: second_task_id=[[_FIRST_EXPLICIT_TASK_ID]],
// CHECK-SAME: prior_task_status=ompt_task_switch=7

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_control_tool:
// CHECK-SAME: command=5, modifier=1, arg=(nil),
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: task level 0:
// CHECK-SAME: task_id=[[_FIRST_EXPLICIT_TASK_ID]]

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: task level 1:
// CHECK-SAME: task_id=[[_FIRST_MASTER_IMPLICIT_TASK_ID]]

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: task level 2:
// CHECK-SAME: task_id=[[_FIRST_INITIAL_TASK_ID]]

// CHECK: {{^}}[[_1ST_MSTR_TID]]:
// CHECK-SAME: _first_tool: parallel level 0: parallel_id=[[_FIRST_PARALLEL_ID]]

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: parallel level 1:
// CHECK-SAME: parallel_id={{[0-9]+}}

// CHECK: {{^}}[[_1ST_MSTR_TID]]:
// CHECK-SAME: _first_tool: ompt_event_task_schedule:
// CHECK-SAME: first_task_id=[[_FIRST_EXPLICIT_TASK_ID]],
// CHECK-SAME: second_task_id=[[_FIRST_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: prior_task_status=ompt_task_complete=1

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_task_end:
// CHECK-SAME: task_id=[[_FIRST_EXPLICIT_TASK_ID]]

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_wait_barrier_end:
// CHECK-SAME: parallel_id=0,
// CHECK-SAME: task_id=[[_FIRST_MASTER_IMPLICIT_TASK_ID]], codeptr_ra=(nil)

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_barrier_end:
// CHECK-SAME: parallel_id=0,
// CHECK-SAME: task_id=[[_FIRST_MASTER_IMPLICIT_TASK_ID]], codeptr_ra=(nil)

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_implicit_task_end:
// CHECK-SAME: parallel_id=0, task_id=[[_FIRST_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: team_size=2, thread_num=0

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_parallel_end:
// CHECK-SAME: parallel_id=[[_FIRST_PARALLEL_ID]],
// CHECK-SAME: task_id=[[_FIRST_INITIAL_TASK_ID]], invoker=2,
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}

// CHECK: {{^}}[[_1ST_MSTR_TID]]: _first_tool: ompt_event_thread_end:
// CHECK-SAME: thread_id=[[_1ST_MSTR_TID]]

// CHECK: {{^}}[[_2ND_MSTR_TID:[0-9]+]]: second_tool: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_initial=1,
// CHECK-SAME: thread_id=[[_2ND_MSTR_TID]]

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_initial_task_begin:
// CHECK-SAME: parallel_id=[[SECOND_INIT_PARALLEL_ID:[0-9]+]],
// CHECK-SAME: task_id=[[SECOND_INITIAL_TASK_ID:[0-9]+]], actual_parallelism=1,
// CHECK-SAME: index=1, flags=1

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_parallel_begin:
// CHECK-SAME: parent_task_id=[[SECOND_INITIAL_TASK_ID]],
// CHECK-SAME: parent_task_frame.exit=(nil),
// CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}},
// CHECK-SAME: parallel_id=[[SECOND_PARALLEL_ID:[0-9]+]], requested_team_size=2,
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}, invoker=2

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_implicit_task_begin:
// CHECK-SAME: parallel_id=[[SECOND_PARALLEL_ID]],
// CHECK-SAME: task_id=[[SECOND_MASTER_IMPLICIT_TASK_ID:[0-9]+]], team_size=2,
// CHECK-SAME: thread_num=0

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_master_begin:
// CHECK-SAME: parallel_id=[[SECOND_PARALLEL_ID]],
// CHECK-SAME: task_id=[[SECOND_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_task_create:
// CHECK-SAME: parent_task_id=[[SECOND_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: parent_task_frame.exit={{0x[0-f]+}},
// CHECK-SAME: parent_task_frame.reenter={{0x[0-f]+}},
// CHECK-SAME: new_task_id=[[SECOND_EXPLICIT_TASK_ID:[0-9]+]],
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}, task_type=ompt_task_explicit=4,
// CHECK-SAME: has_dependences=no

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_master_end:
// CHECK-SAME: parallel_id=[[SECOND_PARALLEL_ID]],
// CHECK-SAME: task_id=[[SECOND_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_barrier_begin:
// CHECK-SAME: parallel_id=[[SECOND_PARALLEL_ID]],
// CHECK-SAME: task_id=[[SECOND_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_wait_barrier_begin:
// CHECK-SAME: parallel_id=[[SECOND_PARALLEL_ID]],
// CHECK-SAME: task_id=[[SECOND_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_task_schedule:
// CHECK-SAME: first_task_id=[[SECOND_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: second_task_id=[[SECOND_EXPLICIT_TASK_ID]],
// CHECK-SAME: prior_task_status=ompt_task_switch=7

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_control_tool:
// CHECK-SAME: command=5, modifier=1, arg=(nil),
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: task level 0:
// CHECK-SAME: task_id=[[SECOND_EXPLICIT_TASK_ID]]

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: task level 1:
// CHECK-SAME: task_id=[[SECOND_MASTER_IMPLICIT_TASK_ID]]

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: task level 2:
// CHECK-SAME: task_id=[[SECOND_INITIAL_TASK_ID]]

// CHECK: {{^}}[[_2ND_MSTR_TID]]:
// CHECK-SAME: second_tool: parallel level 0: parallel_id=[[SECOND_PARALLEL_ID]]

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: parallel level 1:
// CHECK-SAME: parallel_id={{[0-9]+}}

// CHECK: {{^}}[[_2ND_MSTR_TID]]:
// CHECK-SAME: second_tool: ompt_event_task_schedule:
// CHECK-SAME: first_task_id=[[SECOND_EXPLICIT_TASK_ID]],
// CHECK-SAME: second_task_id=[[SECOND_MASTER_IMPLICIT_TASK_ID]],
// CHECK-SAME: prior_task_status=ompt_task_complete=1

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_task_end:
// CHECK-SAME: task_id=[[SECOND_EXPLICIT_TASK_ID]]

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_wait_barrier_end:
// CHECK-SAME: parallel_id=0,
// CHECK-SAME: task_id=[[SECOND_MASTER_IMPLICIT_TASK_ID]], codeptr_ra=(nil)

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_barrier_end:
// CHECK-SAME: parallel_id=0,
// CHECK-SAME: task_id=[[SECOND_MASTER_IMPLICIT_TASK_ID]], codeptr_ra=(nil)

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_implicit_task_end:
// CHECK-SAME: parallel_id=0,
// CHECK-SAME: task_id=[[SECOND_MASTER_IMPLICIT_TASK_ID]], team_size=2,
// CHECK-SAME: thread_num=0

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_parallel_end:
// CHECK-SAME: parallel_id=[[SECOND_PARALLEL_ID]],
// CHECK-SAME: task_id=[[SECOND_INITIAL_TASK_ID]], invoker=2,
// CHECK-SAME: codeptr_ra={{0x[0-f]+}}

// CHECK: {{^}}[[_2ND_MSTR_TID]]: second_tool: ompt_event_thread_end:
// CHECK-SAME: thread_id=[[_2ND_MSTR_TID]]

// CHECK: {{^}}[[_1ST_WRKR_TID:[0-9]+]]: _first_tool: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_worker=2,
// CHECK-SAME: thread_id=[[_1ST_WRKR_TID]]

// CHECK: {{^}}[[_1ST_WRKR_TID]]: _first_tool: ompt_event_implicit_task_begin:
// CHECK-SAME: parallel_id=[[_FIRST_PARALLEL_ID]],
// CHECK-SAME: task_id=[[_FIRST_WORKER_IMPLICIT_TASK_ID:[0-9]+]], team_size=2,
// CHECK-SAME: thread_num=1

// CHECK: {{^}}[[_1ST_WRKR_TID]]: _first_tool: ompt_event_barrier_begin:
// CHECK-SAME: parallel_id=[[_FIRST_PARALLEL_ID]],
// CHECK-SAME: task_id=[[_FIRST_WORKER_IMPLICIT_TASK_ID]], codeptr_ra=(nil)

// CHECK: {{^}}[[_1ST_WRKR_TID]]: _first_tool: ompt_event_wait_barrier_begin:
// CHECK-SAME: parallel_id=[[_FIRST_PARALLEL_ID]],
// CHECK-SAME: task_id=[[_FIRST_WORKER_IMPLICIT_TASK_ID]], codeptr_ra=(nil)

// CHECK: {{^}}[[_1ST_WRKR_TID]]: _first_tool: ompt_event_wait_barrier_end:
// CHECK-SAME: parallel_id=0,
// CHECK-SAME: task_id=[[_FIRST_WORKER_IMPLICIT_TASK_ID]], codeptr_ra=(nil)

// CHECK: {{^}}[[_1ST_WRKR_TID]]: _first_tool: ompt_event_barrier_end:
// CHECK-SAME: parallel_id=0,
// CHECK-SAME: task_id=[[_FIRST_WORKER_IMPLICIT_TASK_ID]], codeptr_ra=(nil)

// CHECK: {{^}}[[_1ST_WRKR_TID]]: _first_tool: ompt_event_implicit_task_end:
// CHECK-SAME: parallel_id=0,
// CHECK-SAME: task_id=[[_FIRST_WORKER_IMPLICIT_TASK_ID]], team_size=0,
// thread_num=1

// CHECK: {{^}}[[_1ST_WRKR_TID]]: _first_tool: ompt_event_thread_end:
// CHECK-SAME: thread_id=[[_1ST_WRKR_TID]]

// CHECK: {{^}}[[_2ND_WRKR_TID:[0-9]+]]: second_tool: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_worker=2,
// CHECK-SAME: thread_id=[[_2ND_WRKR_TID]]

// CHECK: {{^}}[[_2ND_WRKR_TID]]: second_tool: ompt_event_implicit_task_begin:
// CHECK-SAME: parallel_id=[[SECOND_PARALLEL_ID]],
// CHECK-SAME: task_id=[[SECOND_WORKER_IMPLICIT_TASK_ID:[0-9]+]], team_size=2,
// CHECK-SAME: thread_num=1

// CHECK: {{^}}[[_2ND_WRKR_TID]]: second_tool: ompt_event_barrier_begin:
// CHECK-SAME: parallel_id=[[SECOND_PARALLEL_ID]],
// CHECK-SAME: task_id=[[SECOND_WORKER_IMPLICIT_TASK_ID]], codeptr_ra=(nil)

// CHECK: {{^}}[[_2ND_WRKR_TID]]: second_tool: ompt_event_wait_barrier_begin:
// CHECK-SAME: parallel_id=[[SECOND_PARALLEL_ID]],
// CHECK-SAME: task_id=[[SECOND_WORKER_IMPLICIT_TASK_ID]], codeptr_ra=(nil)

// CHECK: {{^}}[[_2ND_WRKR_TID]]: second_tool: ompt_event_wait_barrier_end:
// CHECK-SAME: parallel_id=0,
// CHECK-SAME: task_id=[[SECOND_WORKER_IMPLICIT_TASK_ID]], codeptr_ra=(nil)

// CHECK: {{^}}[[_2ND_WRKR_TID]]: second_tool: ompt_event_barrier_end:
// CHECK-SAME: parallel_id=0,
// CHECK-SAME: task_id=[[SECOND_WORKER_IMPLICIT_TASK_ID]], codeptr_ra=(nil)

// CHECK: {{^}}[[_2ND_WRKR_TID]]: second_tool: ompt_event_implicit_task_end:
// CHECK-SAME: parallel_id=0,
// CHECK-SAME: task_id=[[SECOND_WORKER_IMPLICIT_TASK_ID]], team_size=0,
// CHECK-SAME: thread_num=1

// CHECK: {{^}}[[_2ND_WRKR_TID]]: second_tool: ompt_event_thread_end:
// CHECK-SAME: thread_id=[[_2ND_WRKR_TID]]

#endif /* APP */
