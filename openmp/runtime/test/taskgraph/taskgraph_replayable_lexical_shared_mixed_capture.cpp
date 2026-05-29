// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <cstdio>

__attribute__((noinline)) static int run_taskgraph_mixed_capture(int seed) {
  int x = seed;
  int y = seed * 2;
  int out = -1;
  int fp = 7;

#pragma omp taskgraph graph_id(401)
  {
#pragma omp task replayable(1) shared(x, y, out) firstprivate(fp) depend(inout : x, y)
    {
      x += fp;
      y += x;
      out = y + fp;
    }
  }

  return out;
}

int main() {
  const int first = run_taskgraph_mixed_capture(1);
  const int second = run_taskgraph_mixed_capture(100);

  if (first != 17 || second != 314) {
    std::fprintf(stderr,
                 "FAIL lexical mixed capture replay first=%d second=%d expected=17/314\n",
                 first, second);
    return 1;
  }

  std::fprintf(stderr, "PASS lexical mixed capture replay first=%d second=%d\n",
               first, second);
  return 0;
}

// CHECK: PASS lexical mixed capture replay first=17 second=314
