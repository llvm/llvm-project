// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %not --crash %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <cstdio>

// This seems like it could work in principle, but in general we don't know
// where the targets of the referenced variables are when the task is replayed.
__attribute__((noinline)) static void emit_nonlexical_task(int &x, int &out) {
#pragma omp task replayable(1) shared(x, out) depend(inout : x)
  {
    x += 5;
    out = x;
  }
}

__attribute__((noinline)) static int run_taskgraph_nonlexical(int seed) {
  int x = seed;
  int out = -1;

#pragma omp taskgraph graph_id(312)
  { emit_nonlexical_task(x, out); }

  return out;
}

int main() {
  const int recorded = run_taskgraph_nonlexical(1);
  const int replayed = run_taskgraph_nonlexical(100);

  // The "non-lexical" replayable task is emitted in a helper function outside
  // the taskgraph lexical scope.  We expect this to raise a runtime error.
  if (recorded == replayed) {
    std::fprintf(stderr, "UNEXPECTED SUCCESS nonlexical replay recorded=%d replayed=%d\n",
                 recorded, replayed);
    return 0;
  }

  std::fprintf(stderr,
               "EXPECTED FAILURE nonlexical replay recorded=%d replayed=%d\n",
               recorded, replayed);
  return 1;
}

// CHECK: OMP: Error #302: Cannot locate captured shared variable reference for taskgraph replay
