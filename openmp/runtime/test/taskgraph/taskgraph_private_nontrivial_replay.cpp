// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// Private copies of a type with a non-trivial constructor and destructor, on
// a replayed task and on the tasks of a replayed taskloop.  Each replay
// generates its tasks again (OpenMP 6.0 [14.3]), so each gets a new private
// copy [7.5.3], default-constructed -- or, for a firstprivate,
// copy-constructed from the variable [7.5.4] -- and destroyed when the task
// completes.  A body's changes to its copy never reach a later replay, and
// every copy made is destroyed.

#include <atomic>
#include <cstdio>
#include <vector>

// The copies of the taskloop's tasks are made and destroyed concurrently.
struct Tracked {
  static std::atomic<int> Live;
  std::vector<int> V;
  Tracked() { ++Live; }
  Tracked(const Tracked &O) : V(O.V) { ++Live; }
  ~Tracked() { --Live; }
};
std::atomic<int> Tracked::Live{0};

static size_t TaskSeen[4], LoopSeen[4][2], LoopFirst[4][2];

__attribute__((noinline)) static void run(int R, const Tracked &Init) {
  Tracked P, F = Init;
#pragma omp taskgraph graph_id(3)
  {
#pragma omp task private(P) shared(TaskSeen)
    {
      TaskSeen[R] = P.V.size();
      P.V.push_back(R);
    }
#pragma omp taskloop private(P) firstprivate(F) shared(LoopSeen, LoopFirst)    \
    num_tasks(2)
    for (int I = 0; I < 2; ++I) {
      LoopSeen[R][I] = P.V.size();
      LoopFirst[R][I] = F.V.size();
      P.V.push_back(I);
      F.V.push_back(I);
    }
  }
}

int main() {
#pragma omp parallel
#pragma omp single
  {
    Tracked Init;
    for (int R = 0; R < 4; ++R) {
      Init.V.push_back(R);
      run(R, Init);
    }
  }
  for (int R = 0; R < 4; ++R)
    std::printf("run %d: task %zu, loop %zu %zu, first %zu %zu\n", R,
                TaskSeen[R], LoopSeen[R][0], LoopSeen[R][1], LoopFirst[R][0],
                LoopFirst[R][1]);
  std::printf("live %d\n", Tracked::Live.load());
  return 0;
}

// CHECK:      run 0: task 0, loop 0 0, first 1 1
// CHECK-NEXT: run 1: task 0, loop 0 0, first 2 2
// CHECK-NEXT: run 2: task 0, loop 0 0, first 3 3
// CHECK-NEXT: run 3: task 0, loop 0 0, first 4 4
// CHECK-NEXT: live 0
