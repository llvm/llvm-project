// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt, multicpu
// UNSUPPORTED: gcc
// Compilation fails for icc
// XFAIL: icc
// clang-format on
#include "callback.h"

int main() {
#pragma omp target teams num_teams(2) thread_limit(1)
#pragma omp parallel num_threads(1)
  { printf("In teams parallel\n"); }
  return 0;
}

// clang-format off
// CHECK: 0: NULL_POINTER=[[NULL:.*$]]

// CHECK-NOT: 0: parallel_data initially not null
// CHECK-NOT: 0: task_data initially not null
// CHECK-NOT: 0: thread_data initially not null

// CHECK: {{^}}[[MASTER_0:[0-9]+]]: ompt_event_initial_task_begin:
// CHECK-SAME: task_id=[[INIT_TASK:[0-f]+]], {{.*}}, index=1

// CHECK: {{^}}[[MASTER_0]]: ompt_event_teams_begin:
// CHECK-SAME: parent_task_id=[[INIT_TASK]]
// CHECK-SAME: {{.*}} requested_num_teams=2
// CHECK-SAME: {{.*}} invoker=[[TEAMS_FLAGS:[0-9]+]]

//
// team 0
//
// initial task in the teams construct
// CHECK: {{^}}[[MASTER_0]]: ompt_event_initial_task_begin:
// CHECK-SAME: task_id=[[INIT_TASK_0:[0-f]+]], actual_parallelism=2, index=0

// parallel region forked by runtime
// CHECK: {{^}}[[MASTER_0]]: ompt_event_parallel_begin:
// CHECK-SAME: {{.*}} parent_task_id=[[INIT_TASK_0]]
// CHECK-SAME: {{.*}} parallel_id=[[PAR_0:[0-f]+]]
// CHECK: {{^}}[[MASTER_0]]: ompt_event_implicit_task_begin:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_0]], task_id=[[IMPL_TASK_0:[0-f]+]]

// user parallel region
// CHECK: {{^}}[[MASTER_0]]: ompt_event_parallel_begin:
// CHECK-SAME: {{.*}} parent_task_id=[[IMPL_TASK_0]]
// CHECK-SAME: {{.*}} parallel_id=[[PAR_00:[0-f]+]]
// CHECK: {{^}}[[MASTER_0]]: ompt_event_parallel_end:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_00]], task_id=[[IMPL_TASK_0]]

// CHECK: {{^}}[[MASTER_0]]: ompt_event_implicit_task_end:
// CHECK-SAME: {{.*}} parallel_id={{[0-f]+}}, task_id=[[IMPL_TASK_0]]
// CHECK: {{^}}[[MASTER_0]]: ompt_event_parallel_end:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_0]], task_id=[[INIT_TASK_0]]

// CHECK: {{^}}[[MASTER_0]]: ompt_event_initial_task_end:
// CHECK-SAME: task_id=[[INIT_TASK_0]], actual_parallelism=0, index=0

// CHECK: {{^}}[[MASTER_0]]: ompt_event_teams_end:
// CHECK-SAME: {{.*}} task_id=[[INIT_TASK]], invoker=[[TEAMS_FLAGS]]

// CHECK: {{^}}[[MASTER_0]]: ompt_event_initial_task_end:
// CHECK-SAME: task_id=[[INIT_TASK]], {{.*}}, index=1

//
// team 1
//
// initial task in the teams construct
// CHECK: {{^}}[[MASTER_1:[0-9]+]]: ompt_event_initial_task_begin:
// CHECK-SAME: task_id=[[INIT_TASK_1:[0-f]+]], actual_parallelism=2, index=1

// parallel region forked by runtime
// CHECK: {{^}}[[MASTER_1]]: ompt_event_parallel_begin:
// CHECK-SAME: {{.*}} parent_task_id=[[INIT_TASK_1]]
// CHECK-SAME: {{.*}} parallel_id=[[PAR_ID_1:[0-f]+]]
// CHECK: {{^}}[[MASTER_1]]: ompt_event_implicit_task_begin:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_ID_1]], task_id=[[IMPL_TASK_1:[0-f]+]]

// user parallel region
// CHECK: {{^}}[[MASTER_1]]: ompt_event_parallel_begin:
// CHECK-SAME: {{.*}} parent_task_id=[[IMPL_TASK_1]]
// CHECK-SAME: {{.*}} parallel_id=[[PAR_ID_11:[0-f]+]]
// CHECK: {{^}}[[MASTER_1]]: ompt_event_parallel_end:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_ID_11]], task_id=[[IMPL_TASK_1]]

// CHECK: {{^}}[[MASTER_1]]: ompt_event_implicit_task_end:
// CHECK-SAME: {{.*}} parallel_id={{[0-f]+}}, task_id=[[IMPL_TASK_1]]
// CHECK: {{^}}[[MASTER_1]]: ompt_event_parallel_end:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_ID_1]], task_id=[[INIT_TASK_1]]

// CHECK: {{^}}[[MASTER_1]]: ompt_event_initial_task_end:
// CHECK-SAME: task_id=[[INIT_TASK_1]], actual_parallelism=0, index=1
// clang-format on
