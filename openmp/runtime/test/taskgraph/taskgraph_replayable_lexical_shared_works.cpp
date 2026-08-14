// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <cstdio>

__attribute__((noinline)) static int run_taskgraph_lexical(int seed) {
  int x = seed;
  int out = -1;

#pragma omp taskgraph graph_id(311)
  {
#pragma omp task replayable(1) shared(x, out) depend(inout : x)
    {
      x += 5;
      out = x;
    }
  }

  return out;
}

int main() {
  const int first = run_taskgraph_lexical(1);
  const int second = run_taskgraph_lexical(100);

  if (first != 6 || second != 105) {
    std::fprintf(
        stderr,
        "FAIL lexical shared replay first=%d second=%d expected=6/105\n", first,
        second);
    return 1;
  }

  std::fprintf(stderr, "PASS lexical shared replay first=%d second=%d\n", first,
               second);
  return 0;
}

// CHECK: PASS lexical shared replay first=6 second=105
