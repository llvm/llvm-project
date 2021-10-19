// XFAIL: *
// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt
// UNSUPPORTED: gcc
// Compilation fails for icc
// XFAIL: icc

#include "callback.h"

int main() {
#pragma omp target teams num_teams(1) thread_limit(1)
#pragma omp parallel num_threads(1)
  { printf("In teams\n"); }
  return 0;
}

// CHECK: 0: NULL_POINTER=[[NULL:.*$]]

// CHECK-NOT: 0: parallel_data initially not null
// CHECK-NOT: 0: task_data initially not null
// CHECK-NOT: 0: thread_data initially not null

// CHECK: {{^}}[[MASTER:[0-9]+]]: ompt_event_initial_task_begin:
// CHECK-SAME: task_id=[[INIT_TASK:[0-9]+]], {{.*}}, index=1

// CHECK: {{^}}[[MASTER]]: ompt_event_teams_begin:
// CHECK-SAME: parent_task_id=[[INIT_TASK]]
// CHECK-SAME: {{.*}} requested_num_teams=1
// CHECK-SAME: {{.*}} invoker=[[TEAMS_FLAGS:[0-9]+]]

// initial task in the teams construct starts
// CHECK: {{^}}[[MASTER]]: ompt_event_initial_task_begin:
// CHECK-SAME: task_id=[[INIT_TASK_0:[0-9]+]], actual_parallelism=1, index=0

// parallel region forked by runtime
// CHECK: {{^}}[[MASTER]]: ompt_event_parallel_begin:
// CHECK-SAME: {{.*}} parent_task_id=[[INIT_TASK_0]]
// CHECK-SAME: {{.*}} parallel_id=[[PAR_0:[0-9]+]]
// CHECK: {{^}}[[MASTER]]: ompt_event_implicit_task_begin:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_0]], task_id=[[IMPL_TASK_0:[0-9]+]]

// user parallel region
// CHECK: {{^}}[[MASTER]]: ompt_event_parallel_begin:
// CHECK-SAME: {{.*}} parent_task_id=[[IMPL_TASK_0]]
// CHECK-SAME: {{.*}} parallel_id=[[PAR_00:[0-9]+]]
// CHECK-SAME: {{.*}} requested_team_size=1
// CHECK: {{^}}[[MASTER]]: ompt_event_implicit_task_begin:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_00]], task_id=[[IMPL_TASK_00:[0-9]+]]
// CHECK-SAME: {{.*}} team_size=1, thread_num=0
// CHECK: {{^}}[[MASTER]]: ompt_event_implicit_task_end:
// CHECK-SAME: {{.*}} parallel_id={{[0-9]+}}, task_id=[[IMPL_TASK_00]]
// CHECK: {{^}}[[MASTER]]: ompt_event_parallel_end:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_00]], task_id=[[IMPL_TASK_0]]

// CHECK: {{^}}[[MASTER]]: ompt_event_parallel_end:
// CHECK-SAME: {{.*}} parallel_id=[[PAR_0]], task_id=[[INIT_TASK_0]]

// initial task in the teams construct ends
// CHECK: {{^}}[[MASTER]]: ompt_event_initial_task_end:
// CHECK-SAME: task_id=[[INIT_TASK_0]], actual_parallelism=0, index=0

// CHECK: {{^}}[[MASTER]]: ompt_event_teams_end:
// CHECK-SAME: {{.*}} task_id=[[INIT_TASK]], invoker=[[TEAMS_FLAGS]]

// CHECK: {{^}}[[MASTER]]: ompt_event_initial_task_end:
// CHECK-SAME: task_id=[[INIT_TASK]], {{.*}}, index=1
