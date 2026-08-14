// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// Regression test for a taskgraph deadlock.  Non-lexically scoped tasks
// (without the 'replayable' clause) called while recording a taskgraph would
// previously cause lockups.
// The test should run to completion without crashing.

#include <cstdio>

int x = 0;
int y = 10;
int z = 20;

__attribute__((noinline)) static void workOne() {
#pragma omp task depend(inout : x)
  {}
}

__attribute__((noinline)) static void workTwo() {
#pragma omp task depend(inout : y, z)
  {}
}

__attribute__((noinline)) static void workThree() {
#pragma omp task depend(in : y, z) depend(out : x)
  {}
}

int main() {
#pragma omp parallel
#pragma omp single
  {
    for (int i = 0; i < 10; ++i) {
      // Would deadlock if the tasks are created out-of-line (not inlined) and
      // the recorded graph is replayed on subsequent iterations.
#pragma omp taskgraph
      {
        workOne();
        workTwo();
        workThree();
      }
    }
  }

  std::fprintf(stderr, "PASS taskgraph deadlock repro completed\n");
  return 0;
}

// CHECK: PASS taskgraph deadlock repro completed
