// RUN: %libomptarget-compile-generic -fopenmp-version=60 && \
// RUN: env KMP_TASKGRAPH_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic \
// RUN:     --implicit-check-not='sets:'
// RUN: %libomptarget-compileopt-generic -fopenmp-version=60 && \
// RUN: env KMP_TASKGRAPH_TRACE=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic \
// RUN:     --implicit-check-not='sets:'

// REQUIRES: amdgpu
// REQUIRES: taskgraph

// `mutexinoutset` on a target construct, in the shape where the region builder
// can express the exclusion structurally.  The three targets are twins -- same
// (empty) predecessor and successor sets -- so the merge pass folds them into a
// single EXCLUSIVE region, and __kmp_taskgraph_strip_mutex_sets then drops their
// mutex sets, the region itself being what carries the exclusion from there on.
//
// That strip is what this test is really about, because it is what leaves the
// graph offloadable: the sets reach the plugin through the leaves, and the
// AMDGPU lowering declines any graph whose leaf still carries one (its schedule
// has no way to express "unordered but not simultaneous").  With the sets gone
// the leaves are ordinary, the graph can be claimed, and the backend serializes
// the exclusive region exactly as it does a sequential one -- a total order
// being a trivially correct way to satisfy mutual exclusion.  Hence the implicit
// check-not: no set may survive into the trace.
//
// Whether the plugin goes on to claim the graph is a property of the build
// rather than of the strip, so it is not asserted: a plugin with no taskgraph
// lowering declines every graph (the base finalizeTaskGraph returns UNSUPPORTED)
// and libomp replays it on the host, which is a correct outcome that still has
// to produce the answer below.  The trace is only required to show the graph
// reaching the handoff at all, so that a build which never offers it -- the
// "handoff ABI is unavailable" path -- is still caught.
//
// A target's `mutexinoutset` used to be downgraded to `inout` during recording
// (__kmp_filter_aliased_deps skipped mutex handling for a node with no task,
// which is every target node), which serialized these three in encounter order
// and produced `sequential` here instead.
//
// The buffer is shared and updated non-atomically by all three, so the expected
// value depends on the exclusion holding; `nowait` is what leaves them
// unordered, without which encounter order would order them anyway and the
// exclusive region would never be built.

#include <stdio.h>
#include <stdlib.h>

int main() {
  const int N = 256;
  const int Iters = 3;
  int Mtx;
  int *s = (int *)malloc(N * sizeof(int));
  for (int i = 0; i < N; ++i)
    s[i] = i;

  for (int it = 0; it < Iters; ++it) {
#pragma omp taskgraph
    {
#pragma omp target nowait map(tofrom : s[0 : N]) depend(mutexinoutset : Mtx)
      {
        for (int i = 0; i < N; ++i)
          s[i] += 1;
      }
#pragma omp target nowait map(tofrom : s[0 : N]) depend(mutexinoutset : Mtx)
      {
        for (int i = 0; i < N; ++i)
          s[i] += 2;
      }
#pragma omp target nowait map(tofrom : s[0 : N]) depend(mutexinoutset : Mtx)
      {
        for (int i = 0; i < N; ++i)
          s[i] += 4;
      }
    }
  }

  // Each encounter adds 1 + 2 + 4 in some order, so exactly 7.
  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (s[i] != i + Iters * 7)
      ++errors;

  // CHECK: Processed taskgraph
  // CHECK-NEXT: exclusive {
  // CHECK-NEXT:   target: 0x{{[[:xdigit:]]+}}
  // CHECK-NEXT:   target: 0x{{[[:xdigit:]]+}}
  // CHECK-NEXT:   target: 0x{{[[:xdigit:]]+}}
  // CHECK-NEXT: }
  // CHECK: {{Transmitted taskgraph|was declined by the device plugin}}

  // CHECK: taskgraph target mutex exclusive errors: 0
  printf("taskgraph target mutex exclusive errors: %d\n", errors);
  free(s);
  return errors != 0;
}
