// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// Verifies that a non-trivially-copyable 'firstprivate(saved: ...)' list item
// is copy-constructed, never bitwise-copied: into the saved data environment
// when the task is recorded, and from there into each replay's task.
//
// 'SelfRef' carries an internal pointer that the constructor wires back to
// its own member.  A bitwise copy would leave the copy's internal pointer
// aimed at the object it was copied from -- the recording's copy, which is
// destroyed when the recording's task finishes, or the saved copy -- so a
// replay would read a destroyed object or the wrong one.

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
#pragma omp task firstprivate(saved : Local)                                   \
    shared(observed_via_self, observed_value)
    {
      // Read through the self pointer.  This is exactly the operation that
      // a bitwise copy would break: 'Inside' would point at the object the
      // copy was taken from, not at this copy's 'Value'.
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
      std::fprintf(stderr, "FAIL non-trivial taskgraph copy\n");
      std::_Exit(1);
    }
    std::fprintf(stderr, "PASS non-trivial taskgraph copy\n");
  }
};

static ExitGuard Guard;

int main() {
  // Recording copies Local(7) into the task's '.kmp_privates.t' slot, and
  // from there into the saved data environment.  After this returns 'Local'
  // and the recording's copy are gone, so a saved 'Inside' pointer that still
  // aimed at either would dangle.
  run_taskgraph_selfref(7);

  // Replays.  Each one must observe the saved 7 both directly and through
  // its own copy's self-referencing pointer.  Different seed values are
  // intentionally ignored by the task body: only the saved data environment
  // drives what the task sees.
  for (int i = 0; i < 5; ++i)
    run_taskgraph_selfref(42 + i);

  return 0;
}

// CHECK: PASS non-trivial taskgraph copy
