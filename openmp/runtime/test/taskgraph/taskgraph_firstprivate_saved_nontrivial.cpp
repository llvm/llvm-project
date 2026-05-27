// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// Verifies the constructor/destructor lifecycle of a non-trivially-copyable
// 'firstprivate(saved: ...)' list item on a replayable task.
//
// The saved data environment keeps a copy of the list item, made when the
// task is recorded, for the life of the taskgraph record (OpenMP 6.0 [14.3]).
// Each replay generates the task again, with a new private copy [7.5.3]
// initialized from that saved copy by the copy constructor [7.5.4] and
// destroyed when the task's body ends.  The destructor here writes a
// recognisable sentinel value (-1), so a replay that saw a destroyed saved
// copy would observe -1 rather than the saved value.
//
// Construction/destruction balance is verified at program exit via a
// global guard.

#include <cstdio>
#include <cstdlib>

struct Tracker {
  static int Ctors;
  static int CopyCtors;
  static int Dtors;

  int Value;

  explicit Tracker(int V) : Value(V) { ++Ctors; }
  Tracker(const Tracker &Other) : Value(Other.Value) { ++CopyCtors; }
  // Use a sentinel value on destruction so that any read-after-destroy of
  // the saved snapshot becomes observable.
  ~Tracker() {
    ++Dtors;
    Value = -1;
  }
};

int Tracker::Ctors = 0;
int Tracker::CopyCtors = 0;
int Tracker::Dtors = 0;

// 'Failed' is shared across tasks and threads so we can record a fail
// status from within the task body and across the program-exit guard.
static int Failed = 0;

__attribute__((noinline)) static void run_taskgraph_nontrivial(int seed) {
  Tracker Local(seed);
  int observed = 0;

#pragma omp taskgraph graph_id(927)
  {
#pragma omp task firstprivate(saved : Local) shared(observed)
    {
      // Each replay must observe the value captured at recording time
      // (which was 11), never the sentinel -1 written by
      // Tracker::~Tracker().
      observed = Local.Value;
    }
  }

  if (observed != 11) {
    std::fprintf(stderr, "FAIL replay observed=%d expected=11 seed=%d\n",
                 observed, seed);
    Failed = 1;
  }
}

// Final accounting fires at program exit.  Copies are copy-constructed:
//   1. For the task the recording execution runs, at task allocation, as the
//      initial value its body copies, and destroyed when that task finishes.
//   2. For the record's copy of that task, from (1) when the task is
//      recorded: the saved data environment.  It lives as long as the record,
//      which is freed by __kmp_taskgraph_free; that is not driven by program
//      exit in libomp today, hence the asymmetric expected dtor count below.
//   3. For the recording execution's body, from (1), and destroyed when the
//      body ends.
//   4. For each replay's body, from (2), and destroyed when the body ends.
static const int Replays = 5;
struct ExitGuard {
  ~ExitGuard() {
    bool ok = true;
    if (Tracker::CopyCtors != 3 + Replays) {
      std::fprintf(stderr,
                   "FAIL (exit) CopyCtors=%d expected=%d (two for the "
                   "recording, one for the saved data environment, one per "
                   "replay)\n",
                   Tracker::CopyCtors, 3 + Replays);
      ok = false;
    }
    // Expected destruction count at program exit:
    //   - Local objects: one ctor + one dtor per call to
    //     run_taskgraph_nontrivial.
    //   - The recording execution's two copies and each replay's copy.
    //   - Not the saved copy (see above).
    int expected_dtors = Tracker::Ctors + /* recording */ 2 + Replays;
    if (Tracker::Dtors != expected_dtors) {
      std::fprintf(stderr,
                   "FAIL (exit) ctor/dtor imbalance ctors=%d copyctors=%d "
                   "dtors=%d expected_dtors=%d\n",
                   Tracker::Ctors, Tracker::CopyCtors, Tracker::Dtors,
                   expected_dtors);
      ok = false;
    }

    if (!ok || Failed) {
      std::fprintf(stderr, "FAIL firstprivate(saved) non-trivial lifecycle\n");
      std::_Exit(1);
    }

    std::fprintf(stderr,
                 "PASS firstprivate(saved) non-trivial lifecycle "
                 "ctors=%d copyctors=%d dtors=%d\n",
                 Tracker::Ctors, Tracker::CopyCtors, Tracker::Dtors);
  }
};

// Global guard whose destructor fires at program exit, after all OpenMP
// teardown has occurred and the taskgraph record has been freed.
static ExitGuard Guard;

int main() {
  // Recording run.  Inside the function we create Local(11), which the task's
  // firstprivate copies, and the saved data environment copies that in turn.
  run_taskgraph_nontrivial(11);

  // Replay runs.  Each replay's task copies the saved copy and destroys its
  // own copy when its body ends.  The body observes the saved value (11) on
  // every replay -- the per-call seed argument is intentionally ignored by
  // the task body to demonstrate that the saved data environment drives what
  // the task sees, not the call-site argument.
  for (int i = 0; i < Replays; ++i)
    run_taskgraph_nontrivial(42 + i);

  return 0;
}

// CHECK: PASS firstprivate(saved) non-trivial lifecycle
