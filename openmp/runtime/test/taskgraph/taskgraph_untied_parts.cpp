// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A recorded untied task that reaches a task scheduling point runs in parts,
// and each replay runs all of them, from the first: the part a replay starts
// from is not the one the last replay finished in.

#include <atomic>
#include <cstdio>

static std::atomic<int> Before{0}, After{0};

static void run() {
#pragma omp taskgraph graph_id(1)
  {
    for (int I = 0; I < 4; ++I) {
#pragma omp task untied
      {
        ++Before;
#pragma omp taskyield
        ++After;
      }
    }
  }
}

int main() {
#pragma omp parallel
#pragma omp single
  for (int R = 0; R < 5; ++R)
    run();
  std::printf("before %d after %d\n", Before.load(), After.load());
  return 0;
}

// CHECK: before 20 after 20
