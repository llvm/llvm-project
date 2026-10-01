// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental
// REQUIRES: libomptarget-debug

// Refreshing a by-value firstprivate costs a kernel-argument rewrite only when
// the value actually changed: a replay passing the value the previous one did
// takes the graph as it is.

#include <stdio.h>

static int Out;

static void run(int N) {
#pragma omp taskgraph
  {
#pragma omp target firstprivate(N) map(tofrom : Out)
    Out = N;
  }
}

int main(void) {
  const int N[] = {1, 1, 2, 2};
  int Errors = 0;
  for (int It = 0; It < 4; ++It) {
    fprintf(stderr, "before replay %d\n", It);
    run(N[It]);
    if (Out != N[It]) {
      fprintf(stderr, "replay %d: N %d (want %d)\n", It, Out, N[It]);
      ++Errors;
    }
  }

  // CHECK: before replay 1
  // CHECK-NOT: reparametrizes for a changed literal argument
  // CHECK: before replay 2
  // CHECK: reparametrizes for a changed literal argument
  // CHECK-NOT: reparametrizes for a changed literal argument
  // CHECK-NOT: replay {{[0-9]+}}:
  // CHECK: taskgraph firstprivate reparam errors: 0
  fprintf(stderr, "taskgraph firstprivate reparam errors: %d\n", Errors);
  return Errors != 0;
}
