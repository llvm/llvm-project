// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A taskgraph in a member function, replayed for different objects.  A task
// that reaches the object through `this` must reach the object the function
// was called for this time, which the taskgraph's record holds by value.

#include <cstdio>

struct Counter {
  int Hits = 0;
  __attribute__((noinline)) void run() {
#pragma omp taskgraph graph_id(11)
    {
#pragma omp task
      ++Hits;
    }
  }
};

int main() {
  Counter A, B;
#pragma omp parallel
#pragma omp single
  for (int R = 0; R < 6; ++R)
    ((R & 1) ? B : A).run();
  std::printf("A %d B %d\n", A.Hits, B.Hits);
  return 0;
}

// CHECK: A 3 B 3
