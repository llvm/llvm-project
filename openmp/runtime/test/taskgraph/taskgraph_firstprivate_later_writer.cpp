// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A firstprivate of a non-trivially-copyable type, whose variable a task
// generated after it writes before the first task runs.  The task's copy is
// initialized when the task is generated [7.5.4], and a replay generates the
// recorded tasks in their recorded order (OpenMP 6.0 [14.3]), so on every
// replay as on the recording the copy is the vector before the write -- even
// though the write frees the vector's storage before the reading task's body
// starts.  The later task is ordered after the earlier one's generation, so
// the program has no race.
//
// The gate task holds the reader back until the writer is done.

#include <atomic>
#include <cstdio>
#include <vector>

static std::atomic<int> Written;
static size_t SeenSize[4];
static int SeenFirst[4];
static bool Opened[4];

__attribute__((noinline)) static void run(int R) {
  std::vector<int> V(R + 1, R);
  int Gate = 0;
  Written = 0;
#pragma omp taskgraph graph_id(5)
  {
#pragma omp task depend(out : Gate) shared(Opened)
    {
      // Bounded, so that a schedule that runs the reader first cannot hang.
      for (long I = 0; I < 2000000000L && !Written.load(); ++I)
        ;
      Opened[R] = Written.load();
    }
#pragma omp task depend(in : Gate) firstprivate(V) shared(SeenSize, SeenFirst)
    {
      SeenSize[R] = V.size();
      SeenFirst[R] = V.empty() ? -1 : V[0];
    }
#pragma omp task shared(V)
    {
      V.assign(100, 7);
      Written = 1;
    }
  }
}

int main() {
#pragma omp parallel num_threads(4)
#pragma omp single
  for (int R = 0; R < 4; ++R)
    run(R);
  for (int R = 0; R < 4; ++R)
    std::printf("run %d: size %zu, first %d, writer first %d\n", R, SeenSize[R],
                SeenFirst[R], Opened[R]);
  return 0;
}

// CHECK:      run 0: size 1, first 0, writer first 1
// CHECK-NEXT: run 1: size 2, first 1, writer first 1
// CHECK-NEXT: run 2: size 3, first 2, writer first 1
// CHECK-NEXT: run 3: size 4, first 3, writer first 1
