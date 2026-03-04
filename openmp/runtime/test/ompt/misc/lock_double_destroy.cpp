// clang-format off
// RUN: %libomp-cxx-compile-and-run | FileCheck %s

// disabled until fixed, see: https://github.com/llvm/llvm-project/pull/145625#issuecomment-3007625680
// remove "needs-fix", after fixing the issue in the runtime
// REQUIRES: ompt, needs-fix
// clang-format on
#include "callback.h"
#include "omp_testsuite.h"

// tests that the destructor doesn't segv even though
// ompt_finalize_tool() destroys the lock
struct myLock {
  omp_lock_t lock;
  myLock() { omp_init_lock(&lock); }
  ~myLock() { omp_destroy_lock(&lock); }
};

myLock lock;

int main() {
  go_parallel_nthreads(2);

  printf("Before ompt_finalize_tool\n");
  ompt_finalize_tool();
  printf("After ompt_finalize_tool\n");

  return get_exit_value();
}

// clang-format off
// CHECK: 0: NULL_POINTER=[[NULL:.*$]]
// CHECK: {{^}}[[THREAD_ID:[0-9]+]]: ompt_event_thread_begin:
// CHECK-SAME: thread_type=ompt_thread_initial=1

// CHECK: {{^}}[[THREAD_ID]]: ompt_event_init_lock

// CHECK: {{^}}[[THREAD_ID]]: ompt_event_parallel_begin
// CHECK: {{^}}[[THREAD_ID]]: ompt_event_parallel_end

// CHECK: {{^}}Before ompt_finalize_tool

// CHECK: {{^}}[[THREAD_ID]]: ompt_event_thread_end: thread_id=[[THREAD_ID]]
// CHECK: 0: ompt_event_runtime_shutdown

// CHECK: {{^}}After ompt_finalize_tool
// clang-format on
