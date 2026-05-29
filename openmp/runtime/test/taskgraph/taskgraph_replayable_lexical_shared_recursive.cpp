// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <cstdio>

__attribute__((noinline)) static int run_taskgraph_recursive(int depth, int seed) {
  int x = seed;
  int out = -1;

#pragma omp taskgraph graph_id(450)
  {
#pragma omp task replayable(1) shared(x, out, depth) depend(inout : x)
    {
      x += depth + 1;
      out = x;
    }
  }

  if (depth == 0)
    return out;

  return out + run_taskgraph_recursive(depth - 1, seed + 10);
}

int main() {
  const int first = run_taskgraph_recursive(3, 1);
  const int second = run_taskgraph_recursive(3, 100);

  if (first != 74 || second != 470) {
    std::fprintf(stderr,
                 "FAIL lexical recursive replay first=%d second=%d expected=74/470\n",
                 first, second);
    return 1;
  }

  std::fprintf(stderr, "PASS lexical recursive replay first=%d second=%d\n",
               first, second);
  return 0;
}

// CHECK: PASS lexical recursive replay first=74 second=470
