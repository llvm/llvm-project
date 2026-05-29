// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// Verifies that taskgraph cloning copy-constructs non-trivially-copyable
// 'firstprivate(saved: ...)' list items rather than bitwise-copying them.
//
// 'SelfRef' carries an internal pointer that the constructor wires back to
// its own member.  Any bitwise clone of the task descriptor (the historical
// behaviour) would leave the cloned object's internal pointer aimed at the
// original (which goes out of scope when the recording call returns), so
// the first replay would dereference dangling memory.  When the compiler-
// emitted task-clone helper runs, it copy-constructs the cloned snapshot
// in-place, repairing the internal pointer to refer to the clone's own
// storage.  Subsequent replays therefore observe a consistent value.

#include <cstdio>
#include <cstdlib>

struct SelfRef {
  int Value;
  // Self-pointer that the (copy) constructor steers at our own 'Value'
  // member.  Bitwise duplication leaves this dangling.
  int *Inside;

  explicit SelfRef(int V) : Value(V), Inside(&Value) {}
  SelfRef(const SelfRef &Other) : Value(Other.Value), Inside(&Value) {}
  ~SelfRef() {
    // Poison the self pointer so any dangling read after destruction is
    // observable as a (likely) crash or wrong value.
    Inside = nullptr;
    Value = -1;
  }
};

static int Failed = 0;

__attribute__((noinline)) static void run_taskgraph_selfref(int seed) {
  SelfRef Local(seed);
  int observed_via_self = 0;
  int observed_value = 0;

#pragma omp taskgraph graph_id(3142)
  {
#pragma omp task firstprivate(saved: Local)                                   \
    shared(observed_via_self, observed_value)
    {
      // Read through the self pointer.  This is exactly the operation that
      // bitwise cloning would break: the cloned task's 'Inside' would point
      // at the recording-time stack frame, not at the cloned 'Value'.
      observed_via_self = *Local.Inside;
      observed_value = Local.Value;
    }
  }

  if (observed_value != 7 || observed_via_self != 7) {
    std::fprintf(stderr,
                 "FAIL seed=%d observed_value=%d observed_via_self=%d "
                 "(expected both = 7)\n",
                 seed, observed_value, observed_via_self);
    Failed = 1;
  }
}

struct ExitGuard {
  ~ExitGuard() {
    if (Failed) {
      std::fprintf(stderr, "FAIL non-trivial taskgraph clone\n");
      std::_Exit(1);
    }
    std::fprintf(stderr, "PASS non-trivial taskgraph clone\n");
  }
};

static ExitGuard Guard;

int main() {
  // Recording captures Local(7) into the task's '.kmp_privates.t' slot, then
  // the runtime clones that descriptor.  After this returns 'Local' is gone,
  // so any cloned 'Inside' pointer that still aimed at &Local.Value would
  // dangle.  The clone helper must copy-construct the clone in place to
  // repair the self pointer.
  run_taskgraph_selfref(7);

  // Replays.  Each one must observe the saved 7 both directly and through
  // the (repaired) self-referencing pointer.  Different seed values are
  // intentionally ignored by the task body: only the saved snapshot drives
  // what the task sees.
  for (int i = 0; i < 5; ++i)
    run_taskgraph_selfref(42 + i);

  return 0;
}

// CHECK: PASS non-trivial taskgraph clone
