// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %not --crash %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <cstdio>

__attribute__((noinline)) static int emit_nonlexical_task(int seed) {
  int x = seed;
  int out = -1;

// This is syntactically valid, but a taskgraph replay that includes this
// task cannot possibly succeed, because the stack frame containing 'x' and
// 'out' doesn't exist at replay time.  We can raise a runtime error in that
// case.
// This isn't a compile error because the code is still valid if no taskgraph
// record/replay is in progress.
#pragma omp task replayable(1) shared(x, out) depend(inout : x)
  {
    x += 5;
    out = x;
  }

  return out;
}

__attribute__((noinline)) static int run_taskgraph_nonlexical(int seed) {
  int out;

#pragma omp taskgraph graph_id(312)
  {
#pragma omp task shared(out)
    {
      out = emit_nonlexical_task(seed);
    }
  }

  return out;
}

int main() {
  int out = emit_nonlexical_task(50);
  if (out != 55) {
    std::fprintf(stderr,
                 "UNEXPECTED FAILURE: task outside taskgraph returned %d\n",
                 out);
  }

  const int recorded = run_taskgraph_nonlexical(1);
  const int replayed = run_taskgraph_nonlexical(100);

  // The non-lexical replayable task is emitted in a helper function outside
  // the taskgraph lexical scope.
  if (recorded == replayed) {
    std::fprintf(
        stderr,
        "UNEXPECTED SUCCESS nonlexical replay recorded=%d replayed=%d\n",
        recorded, replayed);
    return 0;
  }

  std::fprintf(stderr,
               "EXPECTED FAILURE nonlexical replay recorded=%d replayed=%d\n",
               recorded, replayed);
  return 1;
}

// CHECK: OMP: Error #302: Cannot locate captured shared variable reference for
// taskgraph replay
