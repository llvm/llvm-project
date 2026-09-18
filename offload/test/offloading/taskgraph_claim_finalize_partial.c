// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=NOGRAPH
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=GRAPH

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental
// Fails until a plugin claims taskgraphs, which the AMDGPU backend is the
// first to do.
// XFAIL: *

// A taskgraph region that is only partly mapped outside the graph when the
// graph is finalized.  Whether the graph keeps a region to itself or borrows
// the program's mapping of it is decided then, once, since the two lower to
// different graphs -- and a region mapped only in part is neither: there is no
// mapping covering it to borrow, and storage of the graph's own would be a
// second device copy of the part that is mapped.  So the graph is declined
// when it is finalized, and every replay runs on the host.
//
// The partial mapping is made by the graph's last host task and undone by its
// first, so that it exists when the graph is finalized and between replays but
// never while the graph's `target` region maps [N / 4, 3 * N / 4), which it
// would otherwise refuse as an extension of mapped data.

#include <omp.h>
#include <stdio.h>

#define N 256
#define ITERS 3

static int A[N];
static int Dev, Associated, Failures;
static void *DevMem;

int main(void) {
  Dev = omp_get_default_device();
  DevMem = omp_target_alloc(N / 2 * sizeof(int), Dev);

  for (int It = 0; It < ITERS; ++It) {
#pragma omp taskgraph
    {
#pragma omp task depend(out : A[0])
      if (Associated) {
        Failures += omp_target_disassociate_ptr(A, Dev) != 0;
        Associated = 0;
      }
#pragma omp target map(tofrom : A[N / 4 : N / 2]) depend(inout : A[0])
      for (int I = N / 4; I < 3 * N / 4; ++I)
        A[I] += 1;
#pragma omp task depend(in : A[0])
      {
        Failures += omp_target_associate_ptr(A, DevMem, N / 2 * sizeof(int), 0,
                                             Dev) != 0;
        Associated = 1;
      }
    }
  }

  if (Associated)
    Failures += omp_target_disassociate_ptr(A, Dev) != 0;
  omp_target_free(DevMem, Dev);

  int Errors = Failures;
  for (int I = 0; I < N; ++I)
    Errors += A[I] != (I >= N / 4 && I < 3 * N / 4 ? ITERS : 0);

  // NOGRAPH-NOT: taskgraph
  // GRAPH: taskgraph {{[0-9]+}} replays on the host: [{{.*}}) is only partly mapped outside the graph
  // GRAPH-NOT: taskgraph

  // NOGRAPH: taskgraph claim finalize partial errors: 0
  // GRAPH: taskgraph claim finalize partial errors: 0
  printf("taskgraph claim finalize partial errors: %d\n", Errors);
  return Errors != 0;
}
