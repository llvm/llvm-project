// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu
// REQUIRES: omp_taskgraph_experimental

// Mapping through a struct member, which is the shape that produces member-of
// and pointer-and-pointee map entries plus the attachments that go with them.
// Nothing about it needs a lowering category of its own: the plan's range
// gathering already puts a member inside the range covering its struct, which
// is exactly the coherence the two need, and the kernel-argument arithmetic
// follows from the same gathering.  What the shape does need is relocation,
// since the base of `s.a[0:N]` is what `s.a` holds and that is loaded afresh on
// every replay.
//
// Both `s.a` and `s.b` are mapped, so the graph carries two attachments over one
// struct; getting either wrong points the kernel at host memory.

#include <stdio.h>
#include <stdlib.h>

struct Buffers {
  int *a;
  int *b;
};

int main() {
  const int N = 1024;
  const int Iters = 3;
  struct Buffers s;
  s.a = (int *)malloc(N * sizeof(int));
  s.b = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i) {
    s.a[i] = i;
    s.b[i] = 2 * i;
  }

  for (int it = 0; it < Iters; ++it) {
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : s.a[0 : N]) map(to : s.b[0 : N])
      {
        for (int i = 0; i < N; ++i)
          s.a[i] = s.a[i] + s.b[i];
      }
    }
  }

  // Each encounter adds b[i] == 2 * i, so a[i] == i + Iters * 2 * i.
  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (s.a[i] != i + Iters * 2 * i)
      ++errors;

  // CHECK: taskgraph struct member errors: 0
  printf("taskgraph struct member errors: %d\n", errors);
  free(s.b);
  free(s.a);
  return errors != 0;
}
