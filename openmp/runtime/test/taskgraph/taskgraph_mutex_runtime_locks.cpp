// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s
// RUN: env KMP_TASKGRAPH_TRACE=1 %libomp-run 2>&1 | FileCheck --check-prefix=SHAPE --implicit-check-not='exclusive' %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// Mutual exclusion for a mutexinoutset that the region builder cannot express
// structurally, so it has to be enforced by a lock during replay.
//
// The mutex sets overlap in a chain rather than partitioning into cliques:
//
//   TA { d0 }      TB { d1 }      TAB { d0, d1 }
//
// TAB conflicts with both TA and TB, but TA and TB do not conflict with each
// other, so no grouping of the three into EXCLUSIVE regions expresses exactly
// this relation -- __kmp_taskgraph_find_exclusive_regions forms none, and
// __kmp_taskgraph_strip_mutex_sets keeps the sets, counting them into
// taskgraph->num_mutexes.  The SHAPE run pins that: no exclusive is formed and
// [sets: ...] survives on the nodes, so this really is the lock path and not
// the structural one covered by taskgraph_mutex_exclusive_serialized.cpp.
//
// The set is duplicated over d2/d3 so the graph matches taskgraph_deps_13.cpp,
// whose trace already records that these sets survive the builder.
//
// Nothing here constrains the order the conflicting tasks run in, only that
// they do not overlap: the recording pass gets that from the normal dependency
// machinery, and replay has to reproduce it via
// __kmp_taskgraph_acquire_locks / __kmp_taskgraph_release_locks.
//
// This does discriminate: suppressing those two calls reports ~100 overlaps out
// of 300 critical sections, and a variant with no dependencies at all reports
// ~160, so the detector is not merely failing to catch a violation.  A team too
// small to run the tasks concurrently makes the check vacuous, but never makes
// it fail.

#include <atomic>
#include <cstdio>
#include <omp.h>

#define ITERS 50
#define NRES 4

// Non-zero only if two tasks holding the same resource were inside their
// critical sections at the same moment.
static std::atomic<int> holders[NRES];
static std::atomic<int> overlaps;

// Stay in the critical section long enough that a concurrent holder is actually
// observed rather than merely possible.  Pass res1 < 0 to hold just one.
static void hold(int res0, int res1) {
  if (holders[res0].fetch_add(1) != 0)
    overlaps.fetch_add(1);
  if (res1 >= 0 && holders[res1].fetch_add(1) != 0)
    overlaps.fetch_add(1);
  double end = omp_get_wtime() + 0.0002;
  while (omp_get_wtime() < end)
    ;
  if (res1 >= 0)
    holders[res1].fetch_sub(1);
  holders[res0].fetch_sub(1);
}

int main() {
  int d[NRES];

  for (int iter = 0; iter < ITERS; ++iter) {
#pragma omp parallel num_threads(4)
#pragma omp single
    {
#pragma omp taskgraph
      {
        // clang-format off
#pragma omp task depend(mutexinoutset : d[0])
        { hold(0, -1); }
#pragma omp task depend(mutexinoutset : d[1])
        { hold(1, -1); }
#pragma omp task depend(mutexinoutset : d[0], d[1])
        { hold(0, 1); }
#pragma omp task depend(mutexinoutset : d[2])
        { hold(2, -1); }
#pragma omp task depend(mutexinoutset : d[3])
        { hold(3, -1); }
#pragma omp task depend(mutexinoutset : d[2], d[3])
        { hold(2, 3); }
        // clang-format on
      }
    }
  }

  std::printf("runtime-lock overlaps: %d\n", overlaps.load());
  return overlaps.load() != 0;
}

// SHAPE: sets:

// CHECK: runtime-lock overlaps: 0
