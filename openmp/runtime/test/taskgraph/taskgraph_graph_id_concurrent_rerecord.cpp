// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_DYNAMIC=FALSE KMP_G_DEBUG=1 %not --crash %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental, libomp_debug
// clang-format on

#include <atomic>
#include <cstdio>
#include <omp.h>

static std::atomic<int> first_thread_inside{0};
static std::atomic<int> release_first_thread{0};

// Deterministically force two threads to hit the same lexical taskgraph
// construct with the same graph_id while the first thread is still recording
// it.  That guarantees the second thread sees an existing record in
// KMP_TDG_RECORDING state and trips the runtime sanity check.
static void enter_same_taskgraph(int tid) {
#pragma omp taskgraph graph_id(17) graph_reset(1)
  {
    if (tid == 0) {
      first_thread_inside.store(1, std::memory_order_release);
      while (release_first_thread.load(std::memory_order_acquire) == 0) {
      }
    }
  }
}

int main() {
#pragma omp parallel num_threads(2)
  {
    const int tid = omp_get_thread_num();

    if (tid == 1) {
      while (first_thread_inside.load(std::memory_order_acquire) == 0) {
      }
    }

    enter_same_taskgraph(tid);

    if (tid == 1)
      release_first_thread.store(1, std::memory_order_release);
  }

  std::fprintf(stderr, "UNEXPECTED SUCCESS\n");
  return 0;
}

// CHECK: *** Multiple threads attempting to re-record taskgraph concurrently:
// CHECK-SAME: graph_id=17
// CHECK: Assertion failure at kmp_tasking.cpp
// CHECK-SAME: old_status == KMP_TDG_READY.
// CHECK: OMP: Error #13: Assertion failure at kmp_tasking.cpp
// CHECK-NOT: UNEXPECTED SUCCESS
