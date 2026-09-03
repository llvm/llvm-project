// RUN: %libomptarget-compilexx-generic -fopenmp-version=60 && \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compilexx-generic -fopenmp-version=60 && \
// RUN:   env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic

// REQUIRES: gpu
// REQUIRES: omp_taskgraph_experimental

// A taskgraph in a member function, replayed for different objects.  A replay
// starts from the taskgraph construct's enclosing data environment (OpenMP 6.0
// [14.3]), where `this` is the object the function was called for this time,
// and a list item named through it -- a member, or a section of one -- is
// reconstructed from it.  So every replay's maps reach this call's object:
// an explicit section of a member array, and the object a target region maps
// implicitly to reach its members.  The same holds for a member mapped by
// `target enter data` and unmapped by `target exit data`.

#include <cstdio>

struct S {
  int M = 0;
  int Out[4] = {0, 0, 0, 0};
  int D[4] = {0, 0, 0, 0};

  __attribute__((noinline)) void run() {
#pragma omp taskgraph graph_id(12)
    {
#pragma omp target map(tofrom : Out[0 : 4])
      Out[0] += 1;
#pragma omp target
      Out[1] += M;
    }
  }

  __attribute__((noinline)) void runData() {
#pragma omp taskgraph graph_id(13)
    {
#pragma omp target enter data map(to : D[0 : 4])
#pragma omp target map(present, alloc : D[0 : 4])
      D[0] += 1;
#pragma omp target exit data map(from : D[0 : 4])
    }
  }
};

int main() {
  S A, B;
  for (int R = 0; R < 4; ++R) {
    S &Obj = (R & 1) ? B : A;
    Obj.M = 10 * R + 1;
    Obj.run();
    // Set on the host between runs, so each run has to copy it in.
    Obj.D[0] = 1000 * R;
    Obj.runData();
  }
  std::printf("A %d %d B %d %d\n", A.Out[0], A.Out[1], B.Out[0], B.Out[1]);
  std::printf("D A %d B %d\n", A.D[0], B.D[0]);
  return 0;
}

// Two runs each; A's M was 1 and 21, B's 11 and 31.
// CHECK: A 2 22 B 2 42
// A's last run was R == 2, B's R == 3.
// CHECK: D A 2001 B 3001
