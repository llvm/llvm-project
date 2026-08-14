// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// Verifies the destructor-lifecycle contract for a non-trivially-copyable
// 'firstprivate(saved: ...)' list item on a replayable task.
//
// The saved snapshot is held in '.kmp_privates.t' of the task descriptor
// that the taskgraph record persists across replays, so the
// compiler-emitted per-task destructor thunk must NOT fire at the end of
// each replay (otherwise subsequent replays would observe a destroyed
// object).  This is verified observationally: the destructor here writes
// a recognisable sentinel value (-1) into the slot, so any replay that
// saw a destroyed snapshot would observe -1 rather than the original
// saved value.
//
// Construction/destruction balance is verified at program exit via a
// global guard whose destructor fires after libomp's static-record
// teardown has run, at which point the saved-firstprivate snapshot must
// also have been destroyed exactly once.

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
      // (which was 11).  If the per-replay destructor deferral were
      // missing, the second and subsequent replays would observe the
      // sentinel -1 written by Tracker::~Tracker().  Do not mutate the
      // snapshot here: we want each replay to read the same value.
      observed = Local.Value;
    }
  }

  if (observed != 11) {
    std::fprintf(stderr, "FAIL replay observed=%d expected=11 seed=%d\n",
                 observed, seed);
    Failed = 1;
  }
}

// Final accounting fires at program exit.  Two snapshots are
// copy-constructed exactly once each at recording time:
//   1. The "original" task's '.kmp_privates.t' snapshot (initialised in IR
//      at task allocation, then destructed when the original task finishes).
//   2. The "clone" task's '.kmp_privates.t' snapshot (initialised by the
//      compiler-emitted task-clone helper invoked from
//      __kmpc_taskgraph_task; its destructor is deferred to
//      __kmp_taskgraph_free, which is not driven by program exit in
//      libomp today, hence the asymmetric expected dtor count below).
struct ExitGuard {
  ~ExitGuard() {
    bool ok = true;
    if (Tracker::CopyCtors != 2) {
      std::fprintf(stderr,
                   "FAIL (exit) CopyCtors=%d expected=2 (one for the "
                   "original task, one for the persistent taskgraph clone)\n",
                   Tracker::CopyCtors);
      ok = false;
    }
    // Expected destruction count at program exit:
    //   - Local objects: one ctor + one dtor per call to
    //     run_taskgraph_nontrivial.
    //   - The original task's snapshot: destructed when the task body
    //     finishes (its task is not taskgraph-owned).
    //   - The clone task's snapshot is deferred to __kmp_taskgraph_free,
    //     which is not yet driven at program exit, so it does not
    //     contribute to the dtor count here.
    int expected_dtors = Tracker::Ctors + /* orig task snapshot */ 1;
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
  // Recording run.  Inside the function we create Local(11), then the saved
  // firstprivate copy-constructs that into the task's slot (one copyctor).
  run_taskgraph_nontrivial(11);

  // Replay runs.  Each replay reuses the recorded task descriptor and its
  // in-place saved snapshot.  No additional copy-construction occurs, and
  // no destruction occurs until end-of-taskgraph.  The body observes the
  // snapshot value (11) on every replay -- the per-call seed argument is
  // intentionally ignored by the task body to demonstrate that the saved
  // snapshot drives what the task sees, not the call-site argument.
  for (int i = 0; i < 5; ++i)
    run_taskgraph_nontrivial(42 + i);

  return 0;
}

// CHECK: PASS firstprivate(saved) non-trivial lifecycle
