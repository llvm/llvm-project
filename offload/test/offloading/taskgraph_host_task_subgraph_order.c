// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env KMP_TASKGRAPH_TRACE=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic --check-prefixes=CHECK,TRACE \
// RUN:   --implicit-check-not='replays on the host'

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// Host tasks that call into the offload runtime themselves, ordered against
// target constructs in the same graph.  Host tasks never enter libomptarget:
// libomp runs them itself, and hands each maximal target-only part of the
// graph to libomptarget as a subgraph of its own, which runs as a single step
// of the replay and has finished when that step returns.  The ordering between
// a host task and a subgraph is therefore libomp's to keep, and that is what
// this tests: every shape checks its data, and a lost edge is a wrong number.
//
// It is a liveness test as well.  Because a subgraph is issued only once the
// host tasks before it have finished, and a host task starts only once the
// subgraphs before it have run, target work is never queued on the device
// behind a host task.  The first two shapes are the ones that hang if that
// ever stops being true: a copy ordered after a host task would stall the
// device's one in-order blit queue, which a copy the task itself issues then
// queues behind; and a host task run as a device-stream callback would run on
// the HSA runtime's async handler thread, which an unpinned device-to-host copy
// from the task then needs.
//
//  * Shape 1: a `target update from`, a subgraph of its own, is ordered after a
//    task that writes the device buffer.
//  * Shape 2: nothing follows the task, but it runs after a subgraph -- a
//    `target update to` and a kernel -- and copies device-to-host.
//  * Shape 3, the join, `seq(par(H1, H2), more)`: two independent tasks, then
//    a subgraph whose kernel names both of them, so it may only run once both
//    have.  The result is only right if the kernel ran after both tasks had
//    finished, not merely started.
//
// The first run turns the AMDGPU graph backend off, so that the plugin claims
// no subgraph and libomp issues each target construct itself; the others let
// the plugin replay the subgraphs, and the traced run checks that it does.
//
// All three are then run again from inside a `parallel` region, which is not
// optional coverage: at top level the team is serialized, so every host task
// is run by the encountering thread.  A task run by another thread, and
// completing while the encountering thread is waiting, is only reached with a
// real team.
//
// The device buffers are allocated and associated with their host arrays
// before any graph runs, so each graph finds them already mapped.  A host task
// therefore gets its device address by capture, and the copy-outs are
// `target update`s.

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 256
#define HALF (N / 2)
#define ITERS 4

static int Dev, Host;

// Run every shape ITERS times over its own buffers, accumulating mismatch
// counts.  Called once at top level and once from a parallel region, so it
// keeps no state of its own beyond the buffers the caller owns.  DevA, DevB
// and DevC are the device buffers associated with A, B and C.
static void runShapes(int *Src, int *A, int *B, int *C, int *Out, int *DevA,
                      int *DevB, int *DevC, int *BadCopyOut, int *BadReadBack,
                      int *BadJoin) {
  int Gate = 0, Gate1 = 0, Gate2 = 0, GateB = 0, GateC = 0;

  for (int It = 0; It < ITERS; ++It) {
    for (int I = 0; I < N; ++I) {
      Src[I] = 100 * (It + 1) + I;
      A[I] = -1;
      B[I] = 1000 * (It + 1) + I;
      C[I] = -1;
      Out[I] = -1;
    }

    // Shape 1: a copy-out is ordered after a host task that writes the device
    // buffer.
#pragma omp taskgraph
    {
#pragma omp task firstprivate(DevA) shared(Src) depend(out : Gate)
      omp_target_memcpy(DevA, Src, N * sizeof(int), 0, 0, Dev, Host);
#pragma omp target update from(A[0 : N]) depend(in : Gate)
    }

    // Shape 2: nothing follows the host task, but it runs after a kernel on
    // the same graph and reads the result back device-to-host.
#pragma omp taskgraph
    {
#pragma omp target update to(B[0 : N]) depend(out : GateB)
#pragma omp target is_device_ptr(DevB) depend(inout : GateB)
      for (int I = 0; I < N; ++I)
        DevB[I] += 1;
#pragma omp task firstprivate(DevB) shared(Out) depend(in : GateB)
      omp_target_memcpy(Out, DevB, N * sizeof(int), 0, 0, Host, Dev);
    }

    // Shape 3: the join.  Two independent host tasks fill one half of the
    // device buffer each, and a kernel that names both adds to the whole of
    // it, so the sum below is only right if the kernel ran after both.
#pragma omp taskgraph
    {
#pragma omp task firstprivate(DevC) shared(Src) depend(out : Gate1)
      omp_target_memcpy(DevC, Src, HALF * sizeof(int), 0, 0, Dev, Host);
#pragma omp task firstprivate(DevC) shared(Src) depend(out : Gate2)
      omp_target_memcpy(DevC, Src, HALF * sizeof(int), HALF * sizeof(int),
                        HALF * sizeof(int), Dev, Host);
#pragma omp target is_device_ptr(DevC) depend(in : Gate1, Gate2)               \
    depend(out : GateC)
      for (int I = 0; I < N; ++I)
        DevC[I] += 7;
#pragma omp target update from(C[0 : N]) depend(in : GateC)
    }

    for (int I = 0; I < N; ++I) {
      if (A[I] != 100 * (It + 1) + I)
        ++*BadCopyOut;
      if (Out[I] != 1000 * (It + 1) + I + 1)
        ++*BadReadBack;
      if (C[I] != 100 * (It + 1) + I + 7)
        ++*BadJoin;
    }
  }
}

int main() {
  Dev = omp_get_default_device();
  Host = omp_get_initial_device();
  int *Src = (int *)malloc(N * sizeof(int));
  int *A = (int *)malloc(N * sizeof(int));
  int *B = (int *)malloc(N * sizeof(int));
  int *C = (int *)malloc(N * sizeof(int));
  int *Out = (int *)malloc(N * sizeof(int));
  int *DevA = (int *)omp_target_alloc(N * sizeof(int), Dev);
  int *DevB = (int *)omp_target_alloc(N * sizeof(int), Dev);
  int *DevC = (int *)omp_target_alloc(N * sizeof(int), Dev);
  omp_target_associate_ptr(A, DevA, N * sizeof(int), 0, Dev);
  omp_target_associate_ptr(B, DevB, N * sizeof(int), 0, Dev);
  omp_target_associate_ptr(C, DevC, N * sizeof(int), 0, Dev);

  int BadCopyOut = 0, BadReadBack = 0, BadJoin = 0;
  runShapes(Src, A, B, C, Out, DevA, DevB, DevC, &BadCopyOut, &BadReadBack,
            &BadJoin);

  // The target constructs go to the device.
  // TRACE: Transmitted taskgraph {{.*}} subgraph
  // CHECK: copy-out did not see the task's writes: 0
  printf("copy-out did not see the task's writes: %d\n", BadCopyOut);
  // CHECK: read-back did not see the kernel's writes: 0
  printf("read-back did not see the kernel's writes: %d\n", BadReadBack);
  // CHECK: joining kernel did not see both tasks' writes: 0
  printf("joining kernel did not see both tasks' writes: %d\n", BadJoin);

  // Again with a team, so the host tasks may run on other threads.  `single`
  // because the graph is per-encounter; the other threads go to its implicit
  // barrier and are available to steal the tasks from there.
  int ParBadCopyOut = 0, ParBadReadBack = 0, ParBadJoin = 0;
#pragma omp parallel num_threads(4)
  {
#pragma omp single
    runShapes(Src, A, B, C, Out, DevA, DevB, DevC, &ParBadCopyOut,
              &ParBadReadBack, &ParBadJoin);
  }

  // CHECK: parallel copy-out did not see the task's writes: 0
  printf("parallel copy-out did not see the task's writes: %d\n",
         ParBadCopyOut);
  // CHECK: parallel read-back did not see the kernel's writes: 0
  printf("parallel read-back did not see the kernel's writes: %d\n",
         ParBadReadBack);
  // CHECK: parallel joining kernel did not see both tasks' writes: 0
  printf("parallel joining kernel did not see both tasks' writes: %d\n",
         ParBadJoin);

  omp_target_disassociate_ptr(C, Dev);
  omp_target_disassociate_ptr(B, Dev);
  omp_target_disassociate_ptr(A, Dev);
  omp_target_free(DevC, Dev);
  omp_target_free(DevB, Dev);
  omp_target_free(DevA, Dev);
  free(Out);
  free(C);
  free(B);
  free(A);
  free(Src);
  return (BadCopyOut + BadReadBack + BadJoin + ParBadCopyOut + ParBadReadBack +
          ParBadJoin) != 0;
}
