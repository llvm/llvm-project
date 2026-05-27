// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A firstprivate of a non-trivially-copyable type on a replayed untied task
// that reaches a task scheduling point.  The body's copy of it, constructed
// from the variable as it is when each replay starts (OpenMP 6.0 [14.3]),
// must last from the task's first part to its last, whichever thread resumes
// it, and every copy made for an encounter is destroyed by the time the
// taskgraph construct completes.

#include <cstdio>
#include <vector>

struct Tracked {
  static int Live;
  std::vector<int> V;
  explicit Tracked(int N) : V(N, 0) { ++Live; }
  Tracked(const Tracked &O) : V(O.V) { ++Live; }
  ~Tracked() { --Live; }
};
int Tracked::Live = 0;

static size_t Before[4], After[4];
static int LiveAfter[4];

__attribute__((noinline)) static void run(int R) {
  Tracked T(R + 1);
#pragma omp taskgraph graph_id(5)
  {
#pragma omp task untied firstprivate(T) shared(Before, After)
    {
      Before[R] = T.V.size();
      T.V.push_back(R);
#pragma omp taskyield
      After[R] = T.V.size();
    }
  }
  // T alone.
  LiveAfter[R] = Tracked::Live;
}

int main() {
#pragma omp parallel
#pragma omp single
  for (int R = 0; R < 4; ++R)
    run(R);
  for (int R = 0; R < 4; ++R)
    std::printf("run %d: before %zu after %zu live %d\n", R, Before[R],
                After[R], LiveAfter[R]);
  return 0;
}

// CHECK:      run 0: before 1 after 2 live 1
// CHECK-NEXT: run 1: before 2 after 3 live 1
// CHECK-NEXT: run 2: before 3 after 4 live 1
// CHECK-NEXT: run 3: before 4 after 5 live 1
