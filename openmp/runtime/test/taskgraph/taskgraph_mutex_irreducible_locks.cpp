// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s
// RUN: env KMP_TASKGRAPH_TRACE=1 %libomp-run 2>&1 | FileCheck --check-prefix=SHAPE --implicit-check-not='exclusive' %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// Surviving mutex sets reached through real dependency edges, rather than the
// flat all-siblings shape of taskgraph_mutex_runtime_locks.cpp.
//
//        T1 (out: a)        T2 (out: b)
//         |        \        /        |
//         |         \      /         |
//      TA{m0}        TAB{m0,m1}      TB{m1}
//         |               |          |
//         '------ T6 (in: x, y, z) --'
//
// The three mutex tasks have *different* predecessor sets (a, b, and both), so
// the twin-merge pass cannot fold them into a single EXCLUSIVE the way it does
// when they are twins, and their sets survive into replay.  The differing
// predecessors also make this shape non-series-parallel, so the members land in
// an IRREDUCIBLE -- which is what the SHAPE run pins, together with the sets
// surviving and no exclusive being formed.
//
// So this covers a lock taken part-way through the descriptor walk rather than
// at the root, and covers it inside a knot: the mutex sets have to survive
// __kmp_taskgraph_gather_mutex_sets descending into the IRREDUCIBLE for the
// bits to be there to lock at all.
//
// The data checks are not incidental -- they confirm the ordering the mutex
// tasks do have (after T1/T2, before T6) is still enforced, so a lock cannot be
// "fixed" by serialising the graph and losing the dependency structure.
//
// This does discriminate: suppressing the two replay-time lock calls reports
// ~100 overlaps here, so a pass is evidence the locks ran rather than evidence
// the detector missed.  A team too small to run the tasks concurrently makes
// the check vacuous, but never makes it fail.

#include <atomic>
#include <cstdio>
#include <omp.h>

#define ITERS 50

// Non-zero only if two tasks whose mutex sets intersect were inside their
// critical sections at the same moment.
static std::atomic<int> holders[2];
static std::atomic<int> overlaps;
static int errors;

// Stay in the critical section long enough that a concurrent holder is actually
// observed rather than merely possible.  Pass r1 < 0 to hold just one.
static void hold(int r0, int r1) {
  if (holders[r0].fetch_add(1) != 0)
    overlaps.fetch_add(1);
  if (r1 >= 0 && holders[r1].fetch_add(1) != 0)
    overlaps.fetch_add(1);
  double end = omp_get_wtime() + 0.0002;
  while (omp_get_wtime() < end)
    ;
  if (r1 >= 0)
    holders[r1].fetch_sub(1);
  holders[r0].fetch_sub(1);
}

int main() {
  int a, b, m0, m1, x, y, z;

  for (int iter = 0; iter < ITERS; ++iter) {
#pragma omp parallel num_threads(4)
#pragma omp single
    {
#pragma omp taskgraph
      {
        // clang-format off
#pragma omp task depend(out : a)
        { a = 1; }
#pragma omp task depend(out : b)
        { b = 2; }
#pragma omp task depend(in : a) depend(mutexinoutset : m0) depend(out : x)
        {
          hold(0, -1);
          x = a;
        }
#pragma omp task depend(in : b) depend(mutexinoutset : m1) depend(out : y)
        {
          hold(1, -1);
          y = b;
        }
#pragma omp task depend(in : a, b) depend(mutexinoutset : m0, m1) depend(out : z)
        {
          hold(0, 1);
          z = a + b;
        }
        // clang-format on
#pragma omp task depend(in : x, y, z)
        {
          if (x != 1 || y != 2 || z != 3)
            ++errors;
        }
      }
    }
  }

  std::printf("irreducible mutex overlaps: %d, errors: %d\n", overlaps.load(),
              errors);
  return overlaps.load() != 0 || errors != 0;
}

// SHAPE: irreducible {
// SHAPE: sets:

// CHECK: irreducible mutex overlaps: 0, errors: 0
