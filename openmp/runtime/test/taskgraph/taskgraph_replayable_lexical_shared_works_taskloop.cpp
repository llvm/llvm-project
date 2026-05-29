// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <cstdio>

__attribute__((noinline)) static int run_taskgraph_lexical(int seed) {
  int x = seed;
  int res = 0;

#pragma omp taskgraph graph_id(611)
  {
#pragma omp taskloop replayable num_tasks(8) shared(x) reduction(+ : res)
    for (int i = 0; i < 16; ++i) {
      res += x + i;
    }
  }

  return res;
}

int main() {
  const int first = run_taskgraph_lexical(1);
  const int second = run_taskgraph_lexical(100);

  if (first != 136 || second != 1720) {
    std::fprintf(stderr,
                 "FAIL lexical shared taskloop replay first=%d second=%d expected=136/1720\n",
                 first, second);
    return 1;
  }

  std::fprintf(stderr,
               "PASS lexical shared taskloop replay first=%d second=%d\n",
               first, second);
  return 0;
}

// CHECK: PASS lexical shared taskloop replay first=136 second=1720
