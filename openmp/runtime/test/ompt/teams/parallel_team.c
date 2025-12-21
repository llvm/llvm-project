// clang-format off
// RUN: %libomp-compile-and-run | %sort-threads | FileCheck %s
// REQUIRES: ompt, multicpu
// UNSUPPORTED: gcc
// Compilation fails for icc
// XFAIL: icc
// clang-format on
#include "callback.h"

int main() {
#pragma omp target teams num_teams(1) thread_limit(2)
#pragma omp parallel num_threads(2)
  { printf("In teams\n"); }
  return 0;
}

// clang-format off
// CHECK: 0: NULL_POINTER=[[NULL:.*$]]

// CHECK-NOT: 0: parallel_data initially not null
// CHECK-NOT: 0: task_data initially not null
// CHECK-NOT: 0: thread_data initially not null

// CHECK: {{^}}[[MASTER:[0-9]+]]: ompt_event_initial_task_begin:
// CHECK-SAME: task_id=[[INIT_TASK:[0-f]+]], {{.*}}, index=1

// CHECK: {{^}}[[MASTER]]: ompt_event_teams_begin:
// CHECK-SAME: parent_task_id=[[INIT_TASK]]
// CHECK-SAME: {{.*}} requested_num_teams=1
// CHECK-SAME: {{.*}} invoker=[[TEAMS_FLAGS:[0-9]+]]

//
// team 0/thread 0
//
// initial task in the teams construct
// CHECK: {{^}}[[MASTER]]: ompt_event_initial_task_begin:
// CHECK-SAME: task_id=[[INIT_TASK_0:[0-f]+]], actual_parallelism=1, index=0

// parallel region forked by runtime
// CHECK: {{^}}[[MASTER]]: ompt_event_parallel_begin:
// CHECK-SAME: {{.*}} parent_task_id=[[INIT_TASK_0]]
// CHECK-SAME: {{.*}} parallel_id=[[PAR_0:[0-f]+]]
// CHECK: {{^}}[[MASTER]]: ompt_event_implicit_task_begin:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_0]], task_id=[[IMPL_TASK_0:[0-f]+]]

// user parallel region
// CHECK: {{^}}[[MASTER]]: ompt_event_parallel_begin:
// CHECK-SAME: {{.*}} parent_task_id=[[IMPL_TASK_0]]
// CHECK-SAME: {{.*}} parallel_id=[[PAR_00:[0-f]+]]
// CHECK-SAME: {{.*}} requested_team_size=2
// CHECK: {{^}}[[MASTER]]: ompt_event_implicit_task_begin:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_00]], task_id=[[IMPL_TASK_00:[0-f]+]]
// CHECK-SAME: {{.*}} team_size=2, thread_num=0
//
// barrier event is here
//
// CHECK: {{^}}[[MASTER]]: ompt_event_implicit_task_end:
// CHECK-SAME: {{.*}} parallel_id={{[0-f]+}}, task_id=[[IMPL_TASK_00]]
// CHECK: {{^}}[[MASTER]]: ompt_event_parallel_end:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_00]], task_id=[[IMPL_TASK_0]]

// CHECK: {{^}}[[MASTER]]: ompt_event_implicit_task_end:
// CHECK-SAME: {{.*}} parallel_id={{[0-f]+}}, task_id=[[IMPL_TASK_0]]
// CHECK: {{^}}[[MASTER]]: ompt_event_parallel_end:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_0]], task_id=[[INIT_TASK_0]]

// CHECK: {{^}}[[MASTER]]: ompt_event_initial_task_end:
// CHECK-SAME: task_id=[[INIT_TASK_0]], actual_parallelism=0, index=0

// CHECK: {{^}}[[MASTER]]: ompt_event_teams_end:
// CHECK-SAME: {{.*}} task_id=[[INIT_TASK]], invoker=[[TEAMS_FLAGS]]

// CHECK: {{^}}[[MASTER]]: ompt_event_initial_task_end:
// CHECK-SAME: task_id=[[INIT_TASK]], {{.*}}, index=1

//
// team 0/thread 1
//
// CHECK: {{^}}[[WORKER:[0-9]+]]: ompt_event_implicit_task_begin:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_00]], task_id=[[IMPL_TASK_01:[0-f]+]]
// CHECK-SAME: {{.*}} team_size=2, thread_num=1
//
// barrier event is here
//
// CHECK: {{^}}[[WORKER]]: ompt_event_implicit_task_end:
// CHECK-SAME: {{.*}} parallel_id={{[0-f]+}}, task_id=[[IMPL_TASK_01]]
// clang-format on
