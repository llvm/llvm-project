// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A replay runs a recorded task from the task descriptor libomp kept when it
// was recorded, so a task's body must not change what the descriptor holds:
// a write to its copy of a firstprivate must not reach a later replay.  Here
// the firstprivate is a variable declared inside the region, which a replay
// takes from the saved data environment (OpenMP 6.0 [14.3]), so every
// encounter sees the value it was recorded with.

#include <cstdio>

static int Seen[4];
static int Encounter;

static void run() {
#pragma omp taskgraph graph_id(1)
  {
    int X = 5;
#pragma omp task firstprivate(X)
    {
      Seen[Encounter] = X;
      X += 100;
    }
  }
  ++Encounter;
}

int main() {
#pragma omp parallel
#pragma omp single
  for (int R = 0; R < 4; ++R)
    run();
  std::printf("%d %d %d %d\n", Seen[0], Seen[1], Seen[2], Seen[3]);
  return 0;
}

// CHECK: 5 5 5 5
