// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A replay starts from the taskgraph construct's enclosing data environment
// (OpenMP 6.0 [14.3]), so a firstprivate of a variable that environment has
// takes the value the variable has at replay -- whether it is a scalar, a
// struct or an array, an automatic variable of the enclosing function or one
// with static storage duration, on a task or on each task of a taskloop.  The
// saved data environment supplies the rest: a list item with the `saved`
// modifier, and a variable declared inside the region, keep the value they had
// when the graph was recorded.

#include <cstdio>

struct Pt {
  int X, Y;
};

static int G;
static int Out[6];
static int Loop[4];

__attribute__((noinline)) static void run(int N, Pt P, int S) {
  int A[3] = {N, N + 1, N + 2};
#pragma omp taskgraph graph_id(7)
  {
    int In = N;
#pragma omp task firstprivate(N, P, A) shared(Out)
    {
      Out[0] = N;
      Out[1] = P.Y;
      Out[2] = A[2];
    }
#pragma omp task firstprivate(G) shared(Out)
    Out[3] = G;
#pragma omp task firstprivate(saved : S) shared(Out)
    Out[4] = S;
#pragma omp task firstprivate(In) shared(Out)
    Out[5] = In;
#pragma omp taskloop firstprivate(N) num_tasks(4) shared(Loop)
    for (int I = 0; I < 4; ++I)
      Loop[I] = N + I;
  }
}

// Replays from Depth frames further down, so the enclosing function's
// variables are not where they were when the graph was recorded.
__attribute__((noinline)) static void deeper(int Depth, int N, Pt P, int S) {
  volatile int Padding[64];
  for (int I = 0; I < 64; ++I)
    Padding[I] = N + I;
  if (Depth)
    deeper(Depth - 1, N, P, S);
  else
    run(N, P, S);
}

int main() {
  int Errors = 0;
#pragma omp parallel num_threads(4)
#pragma omp single
  for (int It = 1; It <= 4; ++It) {
    G = 10 * It;
    deeper(It, It, Pt{0, 20 * It}, 30 * It);
    const int Want[6] = {It, 20 * It, It + 2, 10 * It, 30, 1};
    for (int K = 0; K < 6; ++K)
      if (Out[K] != Want[K]) {
        std::printf("FAIL replay %d Out[%d] %d want %d\n", It, K, Out[K],
                    Want[K]);
        ++Errors;
      }
    for (int K = 0; K < 4; ++K)
      if (Loop[K] != It + K) {
        std::printf("FAIL replay %d Loop[%d] %d want %d\n", It, K, Loop[K],
                    It + K);
        ++Errors;
      }
  }
  std::printf("taskgraph firstprivate refresh errors: %d\n", Errors);
  return Errors != 0;
}

// CHECK-NOT: FAIL
// CHECK: taskgraph firstprivate refresh errors: 0
