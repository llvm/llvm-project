// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A 'replayable' task generated in a function called from the body of a task
// of the taskgraph region.  The body of a task generated in the region is not
// part of the region (OpenMP 6.0 [2], region), so the inner task is not a
// replayable construct of it and is not recorded, though it shares variables
// of a frame that does not exist when the graph is replayed: each replay runs
// the outer task's body again, which generates the inner task afresh, sharing
// that replay's variables.

#include <cstdio>

__attribute__((noinline)) static int emit_nonlexical_task(int seed) {
  int x = seed;
  int out = -1;

#pragma omp task replayable(1) shared(x, out) depend(inout : x)
  {
    x += 5;
    out = x;
  }
#pragma omp taskwait

  return out;
}

__attribute__((noinline)) static int run_taskgraph_nonlexical(int seed) {
  int out;

#pragma omp taskgraph graph_id(312)
  {
#pragma omp task shared(out) firstprivate(seed)
    {
      out = emit_nonlexical_task(seed);
    }
  }

  return out;
}

int main() {
  const int Seeds[] = {1, 100, 1000};
  for (int seed : Seeds)
    std::printf("seed %d: %d\n", seed, run_taskgraph_nonlexical(seed));
  return 0;
}

// CHECK:      seed 1: 6
// CHECK-NEXT: seed 100: 105
// CHECK-NEXT: seed 1000: 1005
