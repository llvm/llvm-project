// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// default(firstprivate) on a replayable task.  Every variable referenced in the
// body without an explicit attribute becomes (plain) firstprivate; here that is
// the automatic array 'local_arr'.  'SavedOff' is captured with the 'saved'
// modifier and 'result' is explicitly shared so the outcome is observable.
//
// Record/replay semantics being verified:
//   - a plain firstprivate is snapshotted at *recording* and reused verbatim on
//     every replay (local_arr keeps its recorded values even though later calls
//     pass a different seed);
//   - firstprivate(saved) likewise keeps the value snapshotted at recording,
//     even after the underlying static is mutated between encounters;
//   - the shared 'result' is relocated to the *current* call's frame on each
//     replay (so replays write to the live object, not the recorded one).
//
// The taskgraph is entered through call_with_depth() at a different recursion
// depth for the recording and for each replay, and clobber_stack() overwrites
// the abandoned frames in between.  This is what makes the relocation clause
// above load-bearing: if every encounter reused a single frame at one address,
// a replay writing through the recorded (stale) shareds pointer would still
// land on the live object by accident and the relocation path would not be
// exercised at all.
//
// 'result' is an int in the frame rather than an int* supplied by the caller so
// that a missing relocation fails as a stale write to a dead frame (leaving
// 'result' at -1) instead of a wild store through a clobbered pointer.
//
// Both firstprivate list items are aggregates (an automatic array and a static
// struct) so the aggregate save path is exercised rather than the
// always-inlined intptr-sized scalar path.

#include <cstdio>

struct Off {
  int a, b, c, d;
};

static Off SavedOff = {1, 2, 3, 4}; // a+b+c+d == 10

static volatile int StackSink = 0;

__attribute__((noinline)) static void clobber_stack(int base) {
  volatile int scratch[4096];

  for (int i = 0; i < 4096; ++i)
    scratch[i] = base + i;

  StackSink += scratch[base & 63];
}

__attribute__((noinline)) static int run_default_firstprivate(int seed) {
  int x = seed; // only used to anchor a dependence
  int local_arr[3] = {seed, seed + 1, seed + 2};
  int result = -1;

#pragma omp taskgraph graph_id(9105)
  {
#pragma omp task default(firstprivate) shared(result)                          \
    firstprivate(saved : SavedOff) depend(inout : x)
    {
      result = local_arr[0] + local_arr[1] + local_arr[2] + SavedOff.a +
               SavedOff.b + SavedOff.c + SavedOff.d;
    }
  }

  return result;
}

__attribute__((noinline)) static int call_with_depth(int seed, int depth) {
  volatile int padding[128];

  for (int i = 0; i < 128; ++i)
    padding[i] = seed + depth + i;

  StackSink += padding[(seed + depth) & 127];

  if (depth == 0)
    return run_default_firstprivate(seed);
  return call_with_depth(seed, depth - 1);
}

int main() {
  bool failed = false;

  // Recording: local_arr == {1,2,3} (sum 6), SavedOff snapshot sum == 10.
  const int a = call_with_depth(1, 0); // 6 + 10
  if (a != 16) {
    std::fprintf(stderr, "FAIL default(firstprivate) record a=%d expected=16\n",
                 a);
    failed = true;
  }

  // Mutate the static; the saved snapshot must keep the recorded value, and the
  // plain firstprivate array must keep its recorded {1,2,3}.  A correct replay
  // therefore still yields 16 -- and it must land in the frame of the current
  // call, which sits at a different stack depth than the recording, proving the
  // shared capture was relocated.
  SavedOff = Off{9, 9, 9, 9};

  clobber_stack(100 * 1000);
  const int b = call_with_depth(100, 3); // frozen: 6 + 10
  clobber_stack(5 * 1000);
  const int c = call_with_depth(5, 6); // frozen: 6 + 10
  if (b != 16 || c != 16) {
    std::fprintf(stderr,
                 "FAIL default(firstprivate) replay b=%d c=%d expected=16/16\n",
                 b, c);
    failed = true;
  }

  if (failed)
    return 1;

  std::fprintf(stderr, "PASS default(firstprivate) task a=%d b=%d c=%d\n", a, b,
               c);
  return 0;
}

// CHECK: PASS default(firstprivate) task a=16 b=16 c=16
