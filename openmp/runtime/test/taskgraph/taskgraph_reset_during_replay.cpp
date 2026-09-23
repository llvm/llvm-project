// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env KMP_TASKGRAPH_CHECK_LIFETIME=1 %libomp-run 2>&1 | FileCheck %s --implicit-check-not='still attached'
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// One thread resets a taskgraph while the others replay it.  A replay holds
// only the record's map_lock and a reset takes only the header lock, so the
// reset used to hand the record to the expire path -- freeing its region tree,
// exec descriptors and recorded clones -- while a replay was still walking
// them.  That crashes or hangs.
//
// Two things this test has to get right, both of which quietly defeat an
// earlier attempt:
//
//   - The reset and the replays must come from the *same* directive site.
//     tdg_handle is per site, so two textual `taskgraph graph_id(1)`
//     directives are separate headers with separate record lists and never
//     interact, no matter how the graph ids line up.  Hence the runtime
//     graph_reset expression below rather than two functions.
//   - The result values prove nothing.  A reset re-records into the record it
//     just freed, so the blocks come straight back off the allocator freelists
//     into structures of the same shape and an in-flight replay reads
//     plausible values; runs that were about to crash still report every
//     result correct.  So the invariant is checked directly, via
//     KMP_TASKGRAPH_CHECK_LIFETIME: the runtime reports any record torn down
//     with a replay still attached, and the implicit-check-not above fails the
//     test if it ever does.  Deferred teardowns are expected and are reported
//     in different words.
//
// The tasks spin so that a replay is in flight long enough for a reset to land
// in the middle of one.

#include <cstdio>
#include <omp.h>

static constexpr int Threads = 16;
static constexpr int Reps = 200;
static constexpr int NTask = 8;
static constexpr long Work = 20000;

static volatile long sink = 0;

static long spin(long n) {
  long acc = 0;
  for (long i = 0; i < n; ++i)
    acc += i ^ n;
  return acc;
}

static void graph(int *out, int base, int reset) {
#pragma omp taskgraph graph_id(1) graph_reset(reset)
  {
    for (int i = 0; i < NTask; ++i) {
#pragma omp task shared(out, base)
      {
        sink += spin(Work);
        out[i] = base + i;
      }
    }
  }
}

int main() {
  int seed[NTask];
  int wrong = 0;

  // Record it once, single-threaded, so that it is READY to replay below.
#pragma omp parallel num_threads(1)
  graph(seed, 0, 0);

#pragma omp parallel num_threads(Threads) reduction(+ : wrong)
  {
    const int me = omp_get_thread_num();
    int out[NTask];
    for (int rep = 0; rep < Reps; ++rep) {
      for (int i = 0; i < NTask; ++i)
        out[i] = -1;
      // Only thread 0 resets; a second resetter would trip the (legitimate)
      // concurrent-re-record diagnostic instead of exercising this path.
      graph(out, me * 100, me == 0);
      for (int i = 0; i < NTask; ++i)
        if (out[i] != me * 100 + i)
          ++wrong;
    }
  }

  // Reported for information only: a reset racing a replay leaves the two
  // encounters unordered, so a replay of the graph as it was before the reset
  // is a legal outcome.  Surviving with the invariant intact is the point.
  std::printf("done, %d unexpected values\n", wrong);
  return 0;
}

// CHECK: done
