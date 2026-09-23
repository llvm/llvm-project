// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A taskgraph inside a parallel region with no enclosing single -- an ordinary,
// spec-legal program -- has every thread encounter the directive and replay the
// same shared record.  Each encounter passes its own outlined-entry args, which
// replay uses to relocate the recorded tasks' by-reference captures onto the
// encountering thread's frame.
//
// map_lock serialises the replay itself: the encountering thread holds it until
// the implicit taskgroup has waited the whole graph out.  The args, though,
// were published before that lock was taken, so a thread queued up behind an
// in-flight replay would retarget relocation at its own frame mid-replay, and
// every task still to be issued would write through to another thread's locals.
// Silent wrong data, no crash.
//
// The chain of dependent tasks matters: a graph whose tasks are all immediately
// runnable is relocated entirely by the encountering thread, whereas a
// successor is relocated from the task-completion hook, on whichever worker
// finished its predecessor.  That is also why the args cannot simply be passed
// down the replay call chain -- relocation does not run on the thread that has
// them.
//
// Racy by nature, so a pass is not a proof; the point is that it reports
// hundreds of wrong values against the unfixed runtime.

#include <cstdio>
#include <omp.h>

static constexpr int Threads = 32;
static constexpr int Reps = 300;
static constexpr int Chain = 6;

int main() {
  int wrong = 0, total = 0;

#pragma omp parallel num_threads(Threads) reduction(+ : wrong, total)
  {
    const int me = omp_get_thread_num();
    // Captured by reference by the recorded tasks, so reaching it on replay
    // depends on relocation having this encounter's args.
    const int local = me * 1000;

    for (int rep = 0; rep < Reps; ++rep) {
      int got[Chain];
      for (int i = 0; i < Chain; ++i)
        got[i] = -1;
      // Serialises the tasks against each other, so that all but the first are
      // released from the completion hook rather than at replay start.
      int chain = 0;

#pragma omp taskgraph
      {
        for (int i = 0; i < Chain; ++i) {
#pragma omp task shared(local, got, chain) depend(inout : chain)
          got[i] = local + i;
        }
      }

      for (int i = 0; i < Chain; ++i) {
        ++total;
        if (got[i] != me * 1000 + i)
          ++wrong;
      }
    }
  }

  if (wrong) {
    std::printf("FAIL %d of %d replayed tasks relocated onto the wrong frame\n",
                wrong, total);
    return 1;
  }
  std::printf("PASS %d replays\n", total);
  return 0;
}

// CHECK: PASS
