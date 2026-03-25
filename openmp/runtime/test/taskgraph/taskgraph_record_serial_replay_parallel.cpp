// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s

// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A graph recorded at top level, where the team is serialized, and replayed
// from inside a parallel region has to run its tasks as tasks of the replaying
// team.  Were the recorded ones to keep the serialized flags they were created
// with, the tasks they create would skip dependence tracking -- so the last
// task below could run before the two it depends on -- and task memory would
// go back to the wrong thread's pool.

#include <stdio.h>

static void shape(int *Out) {
  int G1 = 0, G2 = 0, A = 0, B = 0;
#pragma omp taskgraph
  {
#pragma omp task shared(A, B, G1, G2) firstprivate(Out)
    {
#pragma omp taskgroup
      {
#pragma omp task shared(A) depend(out : G1)
        A = 1;
#pragma omp task shared(B) depend(out : G2)
        B = 2;
#pragma omp task shared(A, B) firstprivate(Out) depend(in : G1, G2)
        *Out += A + B;
      }
    }
  }
}

int main() {
  int Out = 0;
  for (int It = 0; It < 3; ++It)
    shape(&Out);
  // CHECK: serialized: 9
  printf("serialized: %d\n", Out);

  Out = 0;
#pragma omp parallel num_threads(4)
#pragma omp single
  for (int It = 0; It < 3; ++It)
    shape(&Out);
  // CHECK: parallel: 9
  printf("parallel: %d\n", Out);
  return Out != 9;
}
