// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A replay generates its tasks afresh (OpenMP 6.0 [14.3]), each with private
// copies of its own [7.5.3], so a task body that writes its copy of a
// firstprivate affects that replay's task alone.  That holds whatever the copy
// is initialized from: a variable of the taskgraph's enclosing data
// environment, or the saved data environment -- a `saved` list item, or a
// variable declared inside the region -- on a task or on a task of a taskloop.
//
// The replayed task descriptor is the recorded one, so a write that reached
// the descriptor rather than a copy of the replay's own would be seen by the
// next replay, from the second replay on.

#include <cstdio>

static int Out[4][4];

__attribute__((noinline)) static void run(int R, int X, int S) {
#pragma omp taskgraph graph_id(1)
  {
    int L = 30;
#pragma omp task firstprivate(X) shared(Out)
    {
      Out[R][0] = X;
      X += 100;
    }
#pragma omp task firstprivate(saved : S) shared(Out)
    {
      Out[R][1] = S;
      S += 100;
    }
#pragma omp task firstprivate(L) shared(Out)
    {
      Out[R][2] = L;
      L += 100;
    }
#pragma omp taskloop firstprivate(saved : S) shared(Out) num_tasks(1)
    for (int I = 0; I < 1; ++I) {
      Out[R][3] = S;
      S += 100;
    }
  }
}

int main() {
#pragma omp parallel
#pragma omp single
  for (int R = 0; R < 4; ++R)
    run(R, 10, 20 + R);
  for (int R = 0; R < 4; ++R)
    std::printf("run %d: %d %d %d %d\n", R, Out[R][0], Out[R][1], Out[R][2],
                Out[R][3]);
  return 0;
}

// CHECK:      run 0: 10 20 30 20
// CHECK-NEXT: run 1: 10 20 30 20
// CHECK-NEXT: run 2: 10 20 30 20
// CHECK-NEXT: run 3: 10 20 30 20
