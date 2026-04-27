// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A task with a 'replayable' clause in the body of a task that the taskgraph
// records.  The inner construct is encountered by the outer task, whose
// region is not part of the taskgraph region (OpenMP 6.0 [2], region), so it
// is not a replayable construct of the region and is not recorded: a replay
// runs the outer task's body again, which generates the inner task once, from
// the values of that replay.

#include <cstdio>

static int Base;
static int Runs;
static int Seen;

int main() {
#pragma omp parallel num_threads(4)
#pragma omp single
  for (int R = 0; R < 4; ++R) {
    Base = R * 10;
    Runs = 0;
#pragma omp taskgraph
    {
#pragma omp task shared(Runs, Seen)
      {
        int W = Base + 1;
#pragma omp task replayable firstprivate(W) shared(Runs, Seen)
        {
#pragma omp atomic
          ++Runs;
          Seen = W;
        }
#pragma omp taskwait
      }
    }
    std::printf("run %d: inner ran %d time(s), saw %d\n", R, Runs, Seen);
  }
  return 0;
}

// CHECK:      run 0: inner ran 1 time(s), saw 1
// CHECK-NEXT: run 1: inner ran 1 time(s), saw 11
// CHECK-NEXT: run 2: inner ran 1 time(s), saw 21
// CHECK-NEXT: run 3: inner ran 1 time(s), saw 31
