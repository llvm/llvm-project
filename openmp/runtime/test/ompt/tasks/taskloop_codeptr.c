// clang-format off
// RUN: %libomp-compile-and-run | FileCheck %s
// REQUIRES: ompt

// These compilers don't support the taskloop construct
// UNSUPPORTED: gcc-4, gcc-5, icc-16
// GCC 6 has support for taskloops, but at least 6.3.0 is crashing on this test
// UNSUPPORTED: gcc-6
// clang-format on

// Check that the codeptr_ra passed to the work and task_create callbacks of a
// taskloop construct points to the user code and not into the runtime.

#include "callback.h"
#include <omp.h>

int main() {
  unsigned int i, x = 0;

#pragma omp parallel num_threads(1)
  {
#pragma omp taskloop num_tasks(2)
    for (i = 0; i < 4; i++) {
      x++;
    }
    print_fuzzy_address(1);

#pragma omp taskloop num_tasks(2) nogroup
    for (i = 0; i < 4; i++) {
      x++;
    }
    print_fuzzy_address(2);

#pragma omp taskgroup
    {
#pragma omp taskloop num_tasks(2) nogroup
      for (i = 0; i < 4; i++) {
        x++;
      }
      print_fuzzy_address(3);
    }
  }
  // clang-format off
  // CHECK: 0: NULL_POINTER=[[NULL:.*$]]

  // CHECK: {{^}}[[MASTER_ID:[0-9]+]]: ompt_event_implicit_task_begin:
  // taskloop
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskloop_begin:
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS_1:(0x)?[0-f]+]]{{[0-f][0-f]}}, count=4
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS_1]]{{[0-f][0-f]}}
  // CHECK-SAME: task_type=ompt_task_explicit
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS_1]]{{[0-f][0-f]}}
  // CHECK-SAME: task_type=ompt_task_explicit
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskloop_end:
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS_1]]{{[0-f][0-f]}}, count=4
  // CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS_1]]

  // taskloop nogroup
  // CHECK-NOT: {{^}}[[MASTER_ID]]: ompt_event_taskgroup_begin:
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskloop_begin:
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS_2:(0x)?[0-f]+]]{{[0-f][0-f]}}, count=4
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS_2]]{{[0-f][0-f]}}
  // CHECK-SAME: task_type=ompt_task_explicit
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS_2]]{{[0-f][0-f]}}
  // CHECK-SAME: task_type=ompt_task_explicit
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskloop_end:
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS_2]]{{[0-f][0-f]}}, count=4
  // CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS_2]]

  // taskloop nogroup nested in a taskgroup
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskgroup_begin:
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskloop_begin:
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS_3:(0x)?[0-f]+]]{{[0-f][0-f]}}, count=4
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS_3]]{{[0-f][0-f]}}
  // CHECK-SAME: task_type=ompt_task_explicit
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_task_create:
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS_3]]{{[0-f][0-f]}}
  // CHECK-SAME: task_type=ompt_task_explicit
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskloop_end:
  // CHECK-SAME: codeptr_ra=[[RETURN_ADDRESS_3]]{{[0-f][0-f]}}, count=4
  // CHECK: {{^}}[[MASTER_ID]]: fuzzy_address={{.*}}[[RETURN_ADDRESS_3]]
  // CHECK: {{^}}[[MASTER_ID]]: ompt_event_taskgroup_end:
  // clang-format on

  return 0;
}
