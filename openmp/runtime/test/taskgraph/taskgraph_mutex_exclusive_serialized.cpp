// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s
// RUN: env KMP_TASKGRAPH_TRACE=1 %libomp-run 2>&1 | FileCheck --check-prefix=SHAPE --implicit-check-not='sets:' %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// Mutual exclusion for a mutexinoutset that the region builder can express
// structurally.  The two mutex tasks have identical predecessor and successor
// sets, so the twin-merge pass folds them into one EXCLUSIVE region, whose
// children are expanded in sequence:
//
//        T1 (out: d)
//            |
//       [ exclusive ]     both hold { d } as mutexinoutset
//        /         \
//      TA           TB
//            |
//        T4 (in: d)
//
// __kmp_taskgraph_strip_mutex_sets then drops the set (in_exclusive), so replay
// is ordered purely by graph shape and no replay-time lock is involved.  This
// is the control for taskgraph_mutex_runtime_locks.cpp, which covers the shape
// the builder cannot express structurally and which therefore does need locks.
//
// The two paths give the same observable answer, so the SHAPE run is what keeps
// this test honest about which one it took: an exclusive must be formed and no
// [sets: ...] may survive.  Without it, a builder change that stopped forming
// the EXCLUSIVE would leave the sets to be locked at replay instead, and the
// overlap count would stay 0 while the test silently stopped covering the
// structural path.

#include <atomic>
#include <cstdio>
#include <omp.h>

#define ITERS 50

// Number of tasks inside the critical section, and how often that exceeded one.
static std::atomic<int> holders;
static std::atomic<int> overlaps;

// Stay in the critical section long enough that a concurrent holder is actually
// observed rather than merely possible.  A team too small to run the tasks
// concurrently makes this vacuous, but never makes it fail.
static void hold() {
  if (holders.fetch_add(1) != 0)
    overlaps.fetch_add(1);
  double end = omp_get_wtime() + 0.0002;
  while (omp_get_wtime() < end)
    ;
  holders.fetch_sub(1);
}

int main() {
  int d;

  for (int iter = 0; iter < ITERS; ++iter) {
#pragma omp parallel num_threads(4)
#pragma omp single
    {
#pragma omp taskgraph
      {
        // clang-format off
#pragma omp task depend(out : d)
        {}
#pragma omp task depend(mutexinoutset : d)
        { hold(); }
#pragma omp task depend(mutexinoutset : d)
        { hold(); }
#pragma omp task depend(in : d)
        {}
        // clang-format on
      }
    }
  }

  std::printf("exclusive-region overlaps: %d\n", overlaps.load());
  return overlaps.load() != 0;
}

// SHAPE: exclusive {

// CHECK: exclusive-region overlaps: 0
