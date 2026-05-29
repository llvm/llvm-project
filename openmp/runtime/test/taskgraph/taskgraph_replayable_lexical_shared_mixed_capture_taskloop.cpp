// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <cstdio>

__attribute__((noinline)) static int run_taskgraph_mixed_capture(int seed) {
  int x = seed;
  int y = seed * 2;
  int fp = 7;
  int res = 0;

#pragma omp taskgraph graph_id(612)
  {
#pragma omp taskloop replayable num_tasks(8) shared(x, y) firstprivate(fp)     \
    reduction(+ : res)
    for (int i = 0; i < 16; ++i) {
      res += x + y + fp + i;
    }
  }

  return res;
}

int main() {
  const int first = run_taskgraph_mixed_capture(1);
  const int second = run_taskgraph_mixed_capture(100);

  if (first != 280 || second != 5032) {
    std::fprintf(stderr,
                 "FAIL lexical mixed capture taskloop replay first=%d "
                 "second=%d expected=280/5032\n",
                 first, second);
    return 1;
  }

  std::fprintf(
      stderr, "PASS lexical mixed capture taskloop replay first=%d second=%d\n",
      first, second);
  return 0;
}

// CHECK: PASS lexical mixed capture taskloop replay first=280 second=5032
