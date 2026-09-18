// RUN: %libomptarget-compilexx-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_INFO=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental
// Fails until a plugin claims taskgraphs, which the AMDGPU backend is the
// first to do.
// XFAIL: *

// A lambda capturing by reference, called in a target region, is mapped as the
// lambda object plus one fix-up entry per capture: the runtime writes the
// captured variable's device address into the device copy of the lambda, which
// otherwise holds the host address.  The data plan has nothing to perform that
// write with, so a graph containing one must be replayed on the host -- and say
// so -- rather than hand the kernel a lambda full of host pointers.
//
// The fix-up only matters where the target region cannot see the captured
// variables itself, as in the templated forall below: where it can, clang maps
// them as arguments of their own and the kernel uses those.  The lambda is
// static and taken by reference so the graph captures nothing from a stack
// frame, which would otherwise make the backend decline it for relocation and
// hide the missing fix-up.

#include <cstdio>

#define N 256

template <typename BodyTy> __attribute__((noinline)) void forall(BodyTy &Body) {
#pragma omp taskgraph
  {
#pragma omp target teams distribute parallel for
    for (int I = 0; I < N; ++I)
      Body(I);
  }
}

int main() {
  const int Iters = 3;
  double A[N], C[N];
  for (int I = 0; I < N; ++I) {
    A[I] = I;
    C[I] = 0;
  }
  static auto Body = [&](int I) { C[I] += A[I]; };

  // CHECK: taskgraph {{[0-9]+}} replays on the host: argument {{[0-9]+}} of leaf 0 ({{.*}}) is a lambda's by-reference capture
#pragma omp target data map(tofrom : C[0 : N]) map(to : A[0 : N])
  {
    for (int It = 0; It < Iters; ++It)
      forall(Body);
  }

  int Errors = 0;
  for (int I = 0; I < N; ++I)
    if (C[I] != Iters * A[I])
      ++Errors;

  // CHECK: taskgraph lambda capture errors: 0
  std::printf("taskgraph lambda capture errors: %d\n", Errors);
  return Errors != 0;
}
