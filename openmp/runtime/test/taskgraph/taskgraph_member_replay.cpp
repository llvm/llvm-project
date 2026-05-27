// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A taskgraph in a member function, replayed for different objects.  A replay
// starts from the taskgraph construct's enclosing data environment (OpenMP 6.0
// [14.3]), and there `this` is the object the member function was called for
// this time.  So on every replay:
//
//  - a task that reaches the object through `this` reaches this call's object;
//  - a firstprivate of a non-static data member, `firstprivate(m)`, whose list
//    item is `this->m`, takes that member's value as it is now -- whatever its
//    type, on a task or on the tasks of a taskloop;
//  - a `saved` one keeps the value it had when the graph was recorded.

#include <cstdio>
#include <string>

struct S {
  int M = 0;
  std::string Str;
  int Out[8] = {};
  size_t Len[8] = {};
  int Saved[8] = {};
  int Loop[8] = {};

  __attribute__((noinline)) void run(int R) {
#pragma omp taskgraph graph_id(4)
    {
#pragma omp task firstprivate(M, Str) shared(Out, Len)
      {
        Out[R] = M;
        Len[R] = Str.size();
        M = -1;
        Str.clear();
      }
#pragma omp task firstprivate(saved : M) shared(Saved)
      Saved[R] = M;
#pragma omp taskloop firstprivate(M) shared(Loop) num_tasks(1)
      for (int I = 0; I < 1; ++I)
        Loop[R] = M;
    }
  }
};

int main() {
  S A, B;
#pragma omp parallel
#pragma omp single
  for (int R = 0; R < 4; ++R) {
    S &Obj = (R & 1) ? B : A;
    Obj.M = 10 * R + 1;
    Obj.Str.assign(R + 1, 'x');
    Obj.run(R);
  }
  for (int R = 0; R < 4; ++R) {
    S &Obj = (R & 1) ? B : A;
    std::printf("run %d: %d %zu %d %d (M %d)\n", R, Obj.Out[R], Obj.Len[R],
                Obj.Saved[R], Obj.Loop[R], Obj.M);
  }
  return 0;
}

// CHECK:      run 0: 1 1 1 1 (M 21)
// CHECK-NEXT: run 1: 11 2 1 11 (M 31)
// CHECK-NEXT: run 2: 21 3 1 21 (M 21)
// CHECK-NEXT: run 3: 31 4 1 31 (M 31)
