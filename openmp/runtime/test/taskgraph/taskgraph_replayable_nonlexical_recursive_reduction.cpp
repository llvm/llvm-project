// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// Exercises reduction(+: ...) on dynamically (non-lexically) nested
// replayable constructs reached through a recursive driver:
//
//   - emit_reduction_taskloop():  replayable taskloop performing a reduction
//                                 into a file-scope accumulator (Sum).  The
//                                 'seed' parameter is captured as
//                                 firstprivate(saved:) so the snapshot
//                                 taken at recording is reused unchanged
//                                 on every replay (OpenMP 6.0 [7.5.4] /
//                                 [14.3]).
//   - emit_publish_task():        replayable task that reads Sum and stores
//                                 it into a distinct slot of a file-scope
//                                 Snapshots[] array; the slot index is
//                                 carried into the task as a
//                                 firstprivate(saved:) capture so that
//                                 every recursion-level publish lands in
//                                 its own slot and no two publishes race
//                                 for the same destination
//   - driver():                   recursive function that calls
//                                 emit_reduction_taskloop() on descent and
//                                 emit_publish_task() on ascent, so on
//                                 replay these constructs are dispatched
//                                 from differing recursive stack frames
//   - run_taskgraph():            wraps the recursive driver() in
//                                 #pragma omp taskgraph and returns Sum
//                                 (the cumulative reduction, which is the
//                                 deterministic test signal; Snapshots[]
//                                 is written purely as a side-effect to
//                                 exercise the replayable publish tasks)
//
// Per the saved-snapshot semantics, every replay reproduces the recording
// run's reduction value, independent of the seed argument passed to the
// replay invocation.  The test therefore (i) verifies that the recording
// matches expected_result(Seeds[0]) and (ii) verifies that every replay
// matches the recording.  Historically this exercise hit two consecutive
// crashes on replay: first a relocate-side #302 because the (static)
// reduction target's shareds slot couldn't be re-projected from a
// non-OpenMP outer scope, then a taskred-lookup assertion because the
// recording-time taskgroup_t holding the reduction state had been torn
// down.  Both gaps are now fixed in the compiler and runtime.

#include <cstdio>

static constexpr int MaxDepth = 4;

static volatile int Sum = 0;
static volatile int Snapshots[MaxDepth] = {0, 0, 0, 0};

__attribute__((noinline)) static void emit_reduction_taskloop(int seed) {
#pragma omp taskloop replayable num_tasks(8) reduction(+ : Sum)                \
    firstprivate(saved : seed)
  for (int i = 0; i < 16; ++i)
    Sum += seed + i;
}

__attribute__((noinline)) static void emit_publish_task(int slot) {
#pragma omp task replayable firstprivate(saved : slot)
  {
    Snapshots[slot] = Sum;
  }
}

__attribute__((noinline)) static void driver(int depth, int seed) {
  emit_reduction_taskloop(seed + depth);
  if (depth == 0) {
    emit_publish_task(depth);
    return;
  }
  driver(depth - 1, seed);
  emit_publish_task(depth);
}

__attribute__((noinline)) static int run_taskgraph(int seed) {
  Sum = 0;
  for (int i = 0; i < MaxDepth; ++i)
    Snapshots[i] = 0;

#pragma omp taskgraph graph_id(917)
  {
    driver(MaxDepth - 1, seed);
  }

  // Sum is the deterministic test signal: after the taskgraph's implicit
  // taskwait it holds the cumulative reduction across all MaxDepth
  // taskloops.  The per-slot Snapshots[] entries are written by the
  // replayable publish tasks but race with the reduction taskloops, so
  // their individual values are non-deterministic and are intentionally
  // not compared here.  The array exists only so that each replayable
  // publish task targets a distinct destination instead of clobbering a
  // shared scalar.
  return Sum;
}

__attribute__((noinline)) static int expected_result(int seed) {
  int sum = 0;
  for (int d = MaxDepth - 1; d >= 0; --d)
    for (int i = 0; i < 16; ++i)
      sum += seed + d + i;
  return sum;
}

int main() {
  constexpr int NumRuns = 4;
  constexpr int Seeds[NumRuns] = {1, 5, 11, 23};

  int recorded = -1;
  bool failed = false;

#pragma omp parallel num_threads(4)
  {
#pragma omp single
    {
      recorded = run_taskgraph(Seeds[0]);
      const int exp0 = expected_result(Seeds[0]);
      if (recorded != exp0) {
        std::fprintf(stderr, "FAIL initial record got=%d expected=%d\n",
                     recorded, exp0);
        failed = true;
      }

      // Saved-snapshot semantics: every replay should reproduce the
      // recording's reduction value, regardless of the live argument.
      for (int i = 1; i < NumRuns; ++i) {
        const int replayed = run_taskgraph(Seeds[i]);
        if (replayed != recorded) {
          std::fprintf(stderr, "FAIL replay %d seed=%d got=%d recorded=%d\n", i,
                       Seeds[i], replayed, recorded);
          failed = true;
        }
      }
    }
  }

  if (failed)
    return 1;

  std::fprintf(stderr, "PASS non-lexical recursive reduction result=%d\n",
               recorded);
  return 0;
}

// CHECK: PASS non-lexical recursive reduction result=
