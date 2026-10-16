// clang-format off
// RUN: %libomp-compile-and-run 2>&1 | sort | FileCheck %s
// REQUIRES: ompt
// clang-format on

#include <string.h>
#include <stdio.h>
#include "callback.h"

// TODO: use error directive when compiler suppors
typedef void ident_t;
extern void __kmpc_error(ident_t *, int, const char *);

int main() {
#pragma omp parallel num_threads(2)
  {
    if (omp_get_thread_num() == 0) {
      const char *msg = "User message goes here";
      printf("0: Message length=%" PRIu64 "\n", (uint64_t)strlen(msg));
      __kmpc_error(NULL, ompt_warning, msg);
    }
  }
  return 0;
}

// clang-format off
// CHECK: {{^}}0: Message length=[[LENGTH:[0-9]+]]
// CHECK: {{^}}0: NULL_POINTER=[[NULL:.*$]]

// CHECK: {{^}}[[PRIMARY_ID:[0-9]+]]: ompt_event_implicit_task_begin
// CHECK: {{^}}[[PRIMARY_ID]]: ompt_event_runtime_error
// CHECK-SAME: severity=1
// CHECK-SAME: message=User message goes here
// CHECK-SAME: length=[[LENGTH]]
// CHECK-SAME: codeptr_ra={{(0x)?[0-f]+}}

// Message from runtime
// CHECK: {{^}}OMP: Warning{{.*}}User message goes here
// clang-format on
