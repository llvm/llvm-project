// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A replay re-encounters every recorded construct before any of its tasks
// runs (OpenMP 6.0 [14.3]), so a firstprivate takes the value its variable has
// when the replay starts.  A write to the variable by another task of the same
// graph is not seen, even by a task that depends on the writer and so only
// starts once the write has happened.
//
// Only the replays are checked: when the graph is recorded, the writer may
// run before the reader is encountered, and then the reader sees the write.

#include <cstdio>

static int G;
static int Out[2];

__attribute__((noinline)) static void run(int N) {
#pragma omp taskgraph graph_id(11)
  {
#pragma omp task shared(N) depend(out : N)
    {
      N = -1;
      G = -1;
    }
#pragma omp task firstprivate(N, G) shared(Out) depend(in : N)
    {
      Out[0] = N;
      Out[1] = G;
    }
  }
}

int main() {
  int Errors = 0;
#pragma omp parallel num_threads(4)
#pragma omp single
  for (int It = 1; It <= 4; ++It) {
    G = 10 * It;
    run(It);
    if (It == 1)
      continue;
    const int Want[2] = {It, 10 * It};
    for (int K = 0; K < 2; ++K)
      if (Out[K] != Want[K]) {
        std::printf("FAIL replay %d Out[%d] %d want %d\n", It, K, Out[K],
                    Want[K]);
        ++Errors;
      }
  }
  std::printf("taskgraph firstprivate replay start errors: %d\n", Errors);
  return Errors != 0;
}

// CHECK-NOT: FAIL
// CHECK: taskgraph firstprivate replay start errors: 0
