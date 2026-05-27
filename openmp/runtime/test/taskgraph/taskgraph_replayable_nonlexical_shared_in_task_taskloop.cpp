// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// As taskgraph_replayable_nonlexical_shared_in_task.cpp, for a 'replayable'
// taskloop: it is not a replayable construct of the region, so each replay
// runs it afresh, sharing that replay's variables.

#include <cstdio>

__attribute__((noinline)) static int emit_nonlexical_taskloop(int seed) {
  int x = seed;
  int sum = 0;

#pragma omp taskloop replayable num_tasks(8) shared(x) reduction(+ : sum)
  for (int i = 0; i < 16; ++i) {
    sum += x + i;
  }

  return sum;
}

__attribute__((noinline)) static int run_taskgraph_nonlexical(int seed) {
  int out;

#pragma omp taskgraph graph_id(632)
  {
#pragma omp task shared(out) firstprivate(seed)
    {
      out = emit_nonlexical_taskloop(seed);
    }
  }

  return out;
}

int main() {
  // 16 * seed + (0 + ... + 15).
  const int Seeds[] = {1, 100, 1000};
  for (int seed : Seeds)
    std::printf("seed %d: %d\n", seed, run_taskgraph_nonlexical(seed));
  return 0;
}

// CHECK:      seed 1: 136
// CHECK-NEXT: seed 100: 1720
// CHECK-NEXT: seed 1000: 16120
