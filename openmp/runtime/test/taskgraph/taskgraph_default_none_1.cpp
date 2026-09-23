// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A replayable task with default(none) forces every referenced variable to be
// given an explicit data-sharing attribute.  This used to crash Clang's Sema
// (llvm_unreachable "Unexpected clause") because the implicit-DSA check for
// default(none) walked the clause list and had no case for the taskgraph
// 'replayable' clause.  Beyond just compiling, this exercises the two runtime
// mechanisms that must keep working under default(none):
//   - shared-by-reference relocation: the shared 'x'/'out' locals live on a
//     stack frame that moves between the recording call and each replay call,
//     and the shared 'x' must be re-read from the *current* call;
//   - firstprivate(saved) snapshotting of a variable that is mutated between
//     encounters.  A static ARRAY (larger than a pointer) is used on purpose so
//     the aggregate save path -- rather than the always-inlined intptr-sized
//     scalar path -- is exercised.
//
// The frame only moves because each encounter enters the taskgraph through
// call_with_depth() at a different recursion depth, with clobber_stack()
// overwriting the abandoned frames in between.  Called straight from main() the
// frame would recur at one address every time and a replay dereferencing the
// recorded (stale) shareds pointer would still find the live 'x'/'out', so the
// relocation half of the above would go untested.

#include <cstdio>

static int SavedTable[5] = {2, 4, 6, 8, 10}; // sum == 30

static volatile int StackSink = 0;

__attribute__((noinline)) static void clobber_stack(int base) {
  volatile int scratch[4096];

  for (int i = 0; i < 4096; ++i)
    scratch[i] = base + i;

  StackSink += scratch[base & 63];
}

__attribute__((noinline)) static int sum_table(const int *t) {
  int s = 0;
  for (int i = 0; i < 5; ++i)
    s += t[i];
  return s;
}

__attribute__((noinline)) static int run_default_none(int seed) {
  int x = seed;
  int out = -1;

#pragma omp taskgraph graph_id(9101)
  {
#pragma omp task default(none) shared(x, out) firstprivate(saved : SavedTable) \
    depend(inout : x)
    {
      out = x + sum_table(SavedTable);
    }
  }

  return out;
}

__attribute__((noinline)) static int call_with_depth(int seed, int depth) {
  volatile int padding[128];

  for (int i = 0; i < 128; ++i)
    padding[i] = seed + depth + i;

  StackSink += padding[(seed + depth) & 127];

  if (depth == 0)
    return run_default_none(seed);
  return call_with_depth(seed, depth - 1);
}

int main() {
  bool failed = false;

  // Recording: snapshots SavedTable (sum 30) into the task's saved slot.
  const int first = call_with_depth(1, 0); // 1 + 30
  if (first != 31) {
    std::fprintf(stderr, "FAIL default(none) record out=%d expected=31\n",
                 first);
    failed = true;
  }

  // Mutate the whole array; the saved snapshot (sum 30) must keep winning while
  // the shared 'x' is re-read from the current call.
  for (int i = 0; i < 5; ++i)
    SavedTable[i] = 100;

  clobber_stack(50 * 1000);
  const int second = call_with_depth(50, 3); // 50 + 30
  clobber_stack(7 * 1000);
  const int third = call_with_depth(7, 6); // 7 + 30
  if (second != 80 || third != 37) {
    std::fprintf(
        stderr, "FAIL default(none) replay second=%d third=%d expected=80/37\n",
        second, third);
    failed = true;
  }

  if (failed)
    return 1;

  std::fprintf(stderr, "PASS default(none) task first=%d second=%d third=%d\n",
               first, second, third);
  return 0;
}

// CHECK: PASS default(none) task first=31 second=80 third=37
