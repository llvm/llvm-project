// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %not --crash %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <cstdio>

__attribute__((noinline)) static int emit_nonlexical_taskloop(int &x) {
  int sum = 0;
#pragma omp taskloop replayable num_tasks(8) shared(x) reduction(+ : sum)
  for (int i = 0; i < 16; ++i) {
    sum += x + i;
  }
  x += 5;
  return sum;
}

__attribute__((noinline)) static int run_taskgraph_nonlexical(int seed) {
  int x = seed;
  int out = -1;

#pragma omp taskgraph graph_id(631)
  { out = emit_nonlexical_taskloop(x); }

  return out;
}

int main() {
  const int recorded = run_taskgraph_nonlexical(1);
  const int replayed = run_taskgraph_nonlexical(100);

  if (recorded == replayed) {
    std::fprintf(stderr,
                 "UNEXPECTED SUCCESS nonlexical taskloop replay recorded=%d replayed=%d\n",
                 recorded, replayed);
    return 0;
  }

  std::fprintf(stderr,
               "EXPECTED FAILURE nonlexical taskloop replay recorded=%d replayed=%d\n",
               recorded, replayed);
  return 1;
}

// CHECK: OMP: Error #302: Cannot locate captured shared variable reference for taskgraph replay
