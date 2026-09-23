// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// OpenMP 6.0 [14.3]: a task-generating construct encountered in a taskgraph
// construct is a replayable construct of the region "unless otherwise specified
// by the replayable clause", and "a replay execution does not entail execution
// of any code that is part of both the taskgraph region and the encountering
// task region".  Together those mean a replayable(false) construct written
// inside a taskgraph region is encountered once, on the execution that runs the
// region body, and is absent from every replay -- while its replayable
// neighbours run on every execution.
//
// This is the behavioural half of
// clang/test/OpenMP/taskgraph_replayable_false_codegen.cpp, which pins which
// runtime entry point each construct is routed to.

#include <atomic>
#include <cstdio>

static constexpr int NumReps = 4; // one recording execution + three replays
static constexpr int LoopTrips = 5;

static std::atomic<int> Failures{0};

// The dependence-ordering case below wants a variable that is shared, not
// firstprivate by default, in the tasks that carry the depend clauses, and one
// that the taskgraph region does not have to save a copy of: static storage
// duration gives both.
static int DepVar;
static std::atomic<int> OrderViolations{0};

static void expect(const char *What, int Actual, int Expected) {
  if (Actual != Expected) {
    std::fprintf(stderr, "FAIL %s=%d expected=%d\n", What, Actual, Expected);
    ++Failures;
  }
}

// Enough work to make an unsynchronised read of Ordered likely to lose the
// race, so that the taskwait case below is not vacuous.
static void spin() {
  volatile int sink = 0;
  for (int i = 0; i < 2000000; ++i)
    sink += i;
}

int main() {
  std::atomic<int> Replayed{0}, Once{0};
  std::atomic<int> LoopReplayed{0}, LoopOnce{0};
  std::atomic<int> ResetEveryTime{0};
  std::atomic<int> DynFalseAtRecord{0}, Chained{0};

#pragma omp parallel num_threads(4)
#pragma omp single
  {
    for (int rep = 0; rep < NumReps; ++rep) {
#pragma omp taskgraph graph_id(1)
      {
#pragma omp task
        ++Replayed;
#pragma omp task replayable(false)
        ++Once;
#pragma omp taskloop
        for (int i = 0; i < LoopTrips; ++i)
          ++LoopReplayed;
#pragma omp taskloop replayable(false)
        for (int i = 0; i < LoopTrips; ++i)
          ++LoopOnce;
      }
    }

    // Re-recording the graph on every encounter runs the body every time, so
    // the non-replayable task is encountered every time too.  Without this the
    // "runs once" result above could just as well be a task that never ran
    // more than once for some unrelated reason.
    for (int rep = 0; rep < NumReps; ++rep) {
#pragma omp taskgraph graph_id(2) graph_reset
      {
#pragma omp task replayable(false)
        ++ResetEveryTime;
      }
    }

    // A non-constant argument is decided when the construct is encountered,
    // which only happens on the execution that runs the body.  So a condition
    // that is false for the recording execution keeps the construct out of the
    // record for good, whatever it would evaluate to later.
    for (int rep = 0; rep < NumReps; ++rep) {
      const int Replay = rep != 0;
#pragma omp taskgraph graph_id(3)
      {
#pragma omp task replayable(Replay)
        ++DynFalseAtRecord;
      }
    }

    // Dependences between two non-replayable constructs are what the November
    // 2025 errata permits (both sides not replayable), and on the recording
    // execution they are ordered by the ordinary dependence machinery.
    for (int rep = 0; rep < NumReps; ++rep) {
      DepVar = 0;
#pragma omp taskgraph graph_id(4)
      {
#pragma omp task depend(out : DepVar) replayable(false)
        {
          spin();
          DepVar = 42;
        }
#pragma omp task depend(in : DepVar) replayable(false)
        {
          if (DepVar != 42)
            ++OrderViolations;
          ++Chained;
        }
        // A non-replayable taskwait has to wait for real on this execution.
#pragma omp taskwait depend(in : DepVar) replayable(false)
        if (DepVar != 42)
          ++OrderViolations;
      }
    }
  }

  expect("replayed", Replayed.load(), NumReps);
  expect("once", Once.load(), 1);
  expect("loop_replayed", LoopReplayed.load(), NumReps * LoopTrips);
  expect("loop_once", LoopOnce.load(), LoopTrips);
  expect("reset_every_time", ResetEveryTime.load(), NumReps);
  expect("dyn_false_at_record", DynFalseAtRecord.load(), 1);
  expect("chained", Chained.load(), 1);
  expect("order_violations", OrderViolations.load(), 0);

  if (Failures.load())
    return 1;
  std::fprintf(stderr, "PASS replayable(false)\n");
  return 0;
}

// CHECK-NOT: FAIL
// CHECK: PASS replayable(false)
