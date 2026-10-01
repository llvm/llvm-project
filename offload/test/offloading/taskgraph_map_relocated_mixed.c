// RUN: %libomptarget-compile-generic -fopenmp-version=60 -DMODE=0 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=60 -DMODE=0 && \
// RUN: env LIBOMPTARGET_TASKGRAPH_REPORT_DRIFT=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,RELOC
//
// RUN: %libomptarget-compile-generic -fopenmp-version=60 -DMODE=1 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic --check-prefix=HOST1
// RUN: %libomptarget-compile-generic -fopenmp-version=60 -DMODE=1 && \
// RUN: %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic --check-prefixes=OVERLAP,DECLINE,HOST1
//
// RUN: %libomptarget-compile-generic -fopenmp-version=60 -DMODE=2 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic --check-prefix=HOST2
// RUN: %libomptarget-compile-generic -fopenmp-version=60 -DMODE=2 && \
// RUN: %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic --check-prefixes=PARTIAL,DECLINE,HOST2
//
// RUN: %libomptarget-compile-generic -fopenmp-version=60 -DMODE=3 && \
// RUN: env LIBOMPTARGET_AMDGPU_TASKGRAPH=0 \
// RUN:   %libomptarget-run-generic | %fcheck-generic --check-prefix=HOST3
// RUN: %libomptarget-compile-generic -fopenmp-version=60 -DMODE=3 && \
// RUN: env LIBOMPTARGET_TASKGRAPH_REPORT_DRIFT=1 \
// RUN:   %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic --check-prefixes=OVERLAP,DECLINE,AFTER,HOST3

// REQUIRES: gpu
// REQUIRES: amdgpu
// REQUIRES: omp_taskgraph_experimental

// A graph's ranges follow its captures when a replay relocates them, and they
// need not all move by one distance: a pointer parameter's section moves to
// wherever the pointer now points, independently of the frame.  What a range
// may not do is come apart or run into another one, because the grouping is
// fixed when the graph is recorded.
//
//   MODE=0  P and Q move by different distances, and one onto the other's
//           recorded address; the graph follows both.
//   MODE=1  P and Q were separate regions when recorded and name one region
//           on the replay: refused as an overlap.
//   MODE=2  P and Q were one region when recorded and name two on the
//           replay: refused as moving part of a region.
//   MODE=3  MODE=1, then a replay the graph can perform again, which has to
//           reach the device with nothing left over from the declined one.
//
// A refused replay is declined to the host, which re-runs the mappers and so
// handles all three; the results are checked against it.

#include <stdio.h>

#define N 256

static int Buf[3][N];

static void run(int *P, int *Q) {
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : P[0 : N]) map(tofrom : Q[0 : N])
    for (int I = 0; I < N; ++I) {
      P[I] += 1;
      Q[I] += 2;
    }
  }
}

static int uniform(int K) {
  for (int I = 1; I < N; ++I)
    if (Buf[K][I] != Buf[K][0])
      return 0;
  return 1;
}

int main(void) {
#if MODE == 0
  run(Buf[0], Buf[1]);
  run(Buf[2], Buf[0]);
  run(Buf[1], Buf[2]);
#elif MODE == 1
  run(Buf[0], Buf[1]);
  run(Buf[2], Buf[2]);
#elif MODE == 2
  run(Buf[0], Buf[0]);
  run(Buf[1], Buf[2]);
#else
  run(Buf[0], Buf[1]);
  run(Buf[2], Buf[2]);
  run(Buf[1], Buf[2]);
#endif

  // What a refused replay reports, ahead of the results on stdout.
  // OVERLAP: taskgraph relocation put [{{.*}}) over [{{.*}}), which were separate regions when the graph was recorded
  // PARTIAL: taskgraph relocation moved part of the region [{{.*}}) without moving the rest of it
  // DECLINE: taskgraph {{[0-9]+}} replays on the host this time
  // AFTER-NOT: replays on the host
  // AFTER: relocated host captures, re-resolving ranges
  // AFTER-NOT: replays on the host

  // The first call records and the two after it each relocate.
  // RELOC-COUNT-2: relocated host captures, re-resolving ranges
  // RELOC-NOT: relocated host captures

  // CHECK: buffers: 3 3 3 uniform 1
  // HOST1: buffers: 1 2 3 uniform 1
  // HOST2: buffers: 3 1 2 uniform 1
  // HOST3: buffers: 1 3 5 uniform 1
  printf("buffers: %d %d %d uniform %d\n", Buf[0][0], Buf[1][0], Buf[2][0],
         uniform(0) && uniform(1) && uniform(2));
  return 0;
}
