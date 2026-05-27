// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A firstprivate of a non-trivially-copyable type, on a replayed task.  Each
// replay generates the task again (OpenMP 6.0 [14.3]), so the task's copy is
// copy-constructed [7.5.4] from the variable as it is when the replay starts
// -- the replay's enclosing data environment being the taskgraph construct's
// -- and destroyed when the task completes [7.5.3], like any task's.  So:
//
//  - the copy reflects the variable at each replay, not at the recording;
//  - the body's changes to its copy are gone by the next replay;
//  - the copy is destroyed before the taskgraph construct completes, while
//    the frame it was copied from is still live.  Here its destructor reports
//    to a local of that frame;
//  - every copy made is destroyed.
//
// libomp makes more than one copy for an encounter: the task holds the
// variable's value, which its body copies.  So what is checked is that every
// copy made from the frame is destroyed before the construct completes, not
// how many there are.

#include <cstdio>
#include <vector>

struct Counted {
  static int Live;
  int *Copies, *Destroyed;
  std::vector<int> V;
  Counted(int *C, int *D, int N) : Copies(C), Destroyed(D), V(N, 0) { ++Live; }
  Counted(const Counted &O) : Copies(O.Copies), Destroyed(O.Destroyed), V(O.V) {
    ++Live;
    ++*Copies;
  }
  ~Counted() {
    --Live;
    ++*Destroyed;
  }
};
int Counted::Live = 0;

static size_t Seen[4];

__attribute__((noinline)) static bool run(int R) {
  int Copies = 0, Destroyed = 0;
  Counted C(&Copies, &Destroyed, R + 1);
#pragma omp taskgraph graph_id(2)
  {
#pragma omp task firstprivate(C) shared(Seen)
    {
      Seen[R] = C.V.size();
      C.V.push_back(R);
    }
  }
  // C itself is destroyed after this is read.
  return Copies > 0 && Destroyed == Copies;
}

int main() {
  bool AllDestroyed[4];
#pragma omp parallel
#pragma omp single
  for (int R = 0; R < 4; ++R)
    AllDestroyed[R] = run(R);
  for (int R = 0; R < 4; ++R)
    std::printf("run %d: size %zu, copies destroyed %d\n", R, Seen[R],
                AllDestroyed[R]);
  std::printf("live %d\n", Counted::Live);
  return 0;
}

// CHECK:      run 0: size 1, copies destroyed 1
// CHECK-NEXT: run 1: size 2, copies destroyed 1
// CHECK-NEXT: run 2: size 3, copies destroyed 1
// CHECK-NEXT: run 3: size 4, copies destroyed 1
// CHECK-NEXT: live 0
