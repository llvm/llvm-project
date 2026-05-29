// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// XFAIL: *
// clang-format on

#include <cstdio>

__attribute__((noinline)) static int expected_recursive(int depth, int seed) {
  int x = seed;
  int sum_delta = 0;

  for (int i = 0; i < 16; ++i) {
    sum_delta += depth + i + 1;
  }

  x += sum_delta;
  int local = x * 17 + sum_delta;

  if (depth == 0)
    return local;

  return local + expected_recursive(depth - 1, seed + 10);
}

__attribute__((noinline)) static int run_taskgraph_recursive(int depth,
                                                             int seed) {
  int x = seed;
  int *ptr = &x;
  int sum_delta = 0;
  int gid = 615;

#pragma omp taskgraph graph_id(gid)
  {
#pragma omp taskloop replayable num_tasks(8) shared(ptr, depth)                \
    reduction(+ : sum_delta)
    for (int i = 0; i < 16; ++i) {
      int delta = depth + i + 1;
      __atomic_fetch_add(ptr, delta, __ATOMIC_RELAXED);
      sum_delta += delta;
    }
  }

  int local = x * 17 + sum_delta;

  if (depth == 0)
    return local;

  return local + run_taskgraph_recursive(depth - 1, seed + 10);
}

int main() {
  const int first = run_taskgraph_recursive(3, 1);
  const int second = run_taskgraph_recursive(3, 100);
  const int expected_first = expected_recursive(3, 1);
  const int expected_second = expected_recursive(3, 100);

  if (first == expected_first && second == expected_second) {
    std::fprintf(stderr,
                 "UNEXPECTED SUCCESS lexical recursive taskloop replay "
                 "first=%d second=%d expected=%d/%d\n",
                 first, second, expected_first, expected_second);
    return 0;
  }

  std::fprintf(stderr,
               "EXPECTED FAILURE lexical recursive taskloop replay first=%d "
               "second=%d expected=%d/%d\n",
               first, second, expected_first, expected_second);
  return 1;
}

// CHECK: EXPECTED FAILURE lexical recursive taskloop replay
