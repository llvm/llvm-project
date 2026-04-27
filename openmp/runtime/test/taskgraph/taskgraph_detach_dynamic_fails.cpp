// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %not --crash %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// OpenMP 6.0 [14.3]: a detachable task must not be a replayable task in a
// taskgraph region.  The completion event is created when the task is
// generated, and a replay does not generate the task again, so there is no
// event left for the program to fulfil: the node never completes and the
// encounter waits on its taskgroup for ever.  That is what this used to do.
//
// Clang rejects the lexically-nested form outright (see
// clang/test/OpenMP/taskgraph_replayable_restrictions.cpp).  This is the form
// it cannot see: the construct is in a function called from the region, so
// whether it is recorded is only known once the call has been made.

#include <cstdio>
#include <omp.h>

static void make_task(int replayable) {
  omp_event_handle_t ev;
#pragma omp task detach(ev) replayable(replayable)
  {
  }
  omp_fulfill_event(ev);
}

int main() {
#pragma omp parallel num_threads(2)
#pragma omp single
  {
    // Detachable but not replayable: recorded as nothing, runs normally, and
    // must be left alone.
    for (int rep = 0; rep < 2; ++rep) {
#pragma omp taskgraph graph_id(1)
      make_task(0);
    }
    std::fprintf(stderr, "not replayable: ok\n");

    // Replayable: has to be rejected rather than recorded.
#pragma omp taskgraph graph_id(2)
    make_task(1);
  }

  std::fprintf(stderr, "UNEXPECTED SUCCESS\n");
  return 0;
}

// CHECK: not replayable: ok
// CHECK: OMP: Error #{{[0-9]+}}: {{.*}}a detachable task cannot be
// CHECK-SAME: recorded for taskgraph replay
// CHECK-NOT: UNEXPECTED SUCCESS
