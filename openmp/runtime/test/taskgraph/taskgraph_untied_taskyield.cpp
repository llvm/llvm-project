// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// Nothing in the spec stops a recorded task being untied or reaching a task
// scheduling point: untied only says any thread in the binding thread set may
// resume the region after suspension, and the only restriction 6.0 adds to the
// taskgraph construct is that antecedent tasks across constructs agree on being
// replayable.  So both have to work on replay.
//
// They do, and what makes them work is worth pinning down, because it is not
// obvious and it is easy to break.  A recorded untied task really does run in
// several parts, so __kmp_task_finish is reached more than once for the same
// node.  The untied part counter is checked *before* the taskgraph successor
// hook there, so a node releases its successors -- and decrements its taskgroup
// count -- only on its final part.  Release them on an earlier part instead and
// the graph fires successors early and repeatedly; skip the release and the
// graph stalls.
//
// That in turn is what lets __kmp_omp_tg_task zero td_untied_count at the start
// of every replay: map_lock serialises replays, and because the count only
// drops on the last part, the implicit taskgroup wait has drained every part of
// every node before the next replay begins.  Genuinely concurrent replay, or
// honouring nogroup on replay, would zero a live part counter and strand the
// successors of a half-finished node.

#include <cstdio>
#include <omp.h>

static constexpr int Threads = 8;
static constexpr int Reps = 20;
static constexpr int NTask = 4;

int main() {
  int errors = 0;

  // 1. One encounter at a time.  Untied tasks with a task scheduling point in
  //    them must still all run, on every replay.
#pragma omp parallel num_threads(Threads)
  {
#pragma omp single
    {
      for (int rep = 0; rep < Reps; ++rep) {
        int c = 0;
#pragma omp taskgraph
        {
          for (int i = 0; i < NTask; ++i) {
#pragma omp task untied shared(c)
            {
#pragma omp taskyield
#pragma omp atomic
              c++;
            }
          }
        }
        if (c != NTask) {
          std::printf("single: rep %d ran %d of %d tasks\n", rep, c, NTask);
          ++errors;
        }
      }
    }
  }

  // 2. A dependent chain, so that each node's successor release has to happen
  //    exactly once and in order, on the last part of a multi-part task.
#pragma omp parallel num_threads(Threads)
  {
#pragma omp single
    {
      for (int rep = 0; rep < Reps; ++rep) {
        int seq = 0, out_of_order = 0;
#pragma omp taskgraph
        {
          for (int i = 0; i < NTask; ++i) {
#pragma omp task untied shared(seq, out_of_order) depend(inout : seq)
            {
#pragma omp taskyield
              if (seq != i)
                ++out_of_order;
              seq++;
            }
          }
        }
        if (seq != NTask || out_of_order) {
          std::printf("chain: rep %d reached %d of %d, %d out of order\n", rep,
                      seq, NTask, out_of_order);
          ++errors;
        }
      }
    }
  }

  // 3. No single: every thread encounters the directive and replays the shared
  //    record, each relocating the recorded tasks onto its own frame.  A thread
  //    coming up short here means work landed on another thread's frame.
  int short_threads = 0, grand = 0;
#pragma omp parallel num_threads(Threads) reduction(+ : short_threads, grand)
  {
    int mine = 0;
    for (int rep = 0; rep < Reps; ++rep) {
      int c = 0;
#pragma omp taskgraph
      {
        for (int i = 0; i < NTask; ++i) {
#pragma omp task untied shared(c)
          {
#pragma omp taskyield
#pragma omp atomic
            c++;
          }
        }
      }
      mine += c;
    }
    grand += mine;
    if (mine != Reps * NTask)
      ++short_threads;
  }
  if (short_threads) {
    std::printf("unsynchronised: %d of %d threads short, total %d of %d\n",
                short_threads, Threads, grand, Reps * Threads * NTask);
    ++errors;
  }

  std::printf("%s\n", errors ? "FAIL" : "PASS");
  return errors != 0;
}

// CHECK-NOT: {{short|out of order|ran [0-9]+ of|reached [0-9]+ of}}
// CHECK: PASS
