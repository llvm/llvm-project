// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A reduction on a construct that turns out not to be a replayable construct.
//
// While a taskgraph is recording, the reduction init hands its input data to
// the runtime to stash on the taskgroup, so that the next node recorded in that
// taskgroup can adopt it and re-create the reduction state on each replay.  A
// construct that is not replayable records no node, so nothing adopts the
// stash.  Two cases, and they are handled in different places:
//
//   - replayable(false): the compiler can see the construct will take the
//     ordinary path, so it routes the reduction to the plain init and no stash
//     is created at all.
//   - replayable(non-constant): the compiler cannot, so the stash is created
//     and the runtime discards it when the taskgroup ends.  This used to trip
//     an assertion there (and leak in a build without assertions).
//
// Either way the reduction itself must produce the right answer on the
// execution that runs the region body, and the taskloop must be absent from
// every replay.

#include <atomic>
#include <cstdio>

static constexpr int NumReps = 3; // one recording execution + two replays
static constexpr int LoopTrips = 16;

static volatile int Sum;
static std::atomic<int> Failures{0};

static int expected_sum() {
  int Sum = 0;
  for (int i = 0; i < LoopTrips; ++i)
    Sum += i;
  return Sum;
}

static void expect(const char *What, int Rep, int Actual, int Expected) {
  if (Actual != Expected) {
    std::fprintf(stderr, "FAIL %s rep=%d got=%d expected=%d\n", What, Rep,
                 Actual, Expected);
    ++Failures;
  }
}

int main() {
  const int Expected = expected_sum();
  int NotReplayable = 0;

#pragma omp parallel num_threads(4)
#pragma omp single
  {
    for (int rep = 0; rep < NumReps; ++rep) {
      Sum = 0;
#pragma omp taskgraph graph_id(1)
      {
#pragma omp taskloop replayable(false) num_tasks(4) reduction(+ : Sum)
        for (int i = 0; i < LoopTrips; ++i)
          Sum += i;
      }
      // The body only runs on the recording execution, so only that one
      // reduces; a replay has no taskloop to run and leaves Sum alone.
      expect("constant", rep, Sum, rep == 0 ? Expected : 0);
    }

    for (int rep = 0; rep < NumReps; ++rep) {
      Sum = 0;
#pragma omp taskgraph graph_id(2)
      {
#pragma omp taskloop replayable(NotReplayable) num_tasks(4) reduction(+ : Sum)
        for (int i = 0; i < LoopTrips; ++i)
          Sum += i;
      }
      expect("dynamic", rep, Sum, rep == 0 ? Expected : 0);
    }
  }

  if (Failures.load())
    return 1;
  std::fprintf(stderr, "PASS replayable(false) reduction sum=%d\n", Expected);
  return 0;
}

// CHECK-NOT: FAIL
// CHECK: PASS replayable(false) reduction sum=
