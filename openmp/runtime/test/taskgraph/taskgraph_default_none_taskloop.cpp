// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// default(none) on a replayable taskloop.  As with the task variant this used
// to crash Sema.  The loop iteration variable is predetermined private and so
// needs no explicit clause; everything else is spelled out.  The shared 'x'
// local is relocated (and re-read) between the recording call and the replays,
// 'res' is a reduction, and 'SavedBias' is snapshotted with firstprivate(saved)
// and must survive being mutated between encounters.  An aggregate (struct) is
// used for the saved list item so the aggregate save path is exercised rather
// than the always-inlined intptr-sized scalar path.
//
// Each encounter enters the taskgraph through call_with_depth() at a different
// recursion depth, with clobber_stack() overwriting the abandoned frames in
// between, so the frame holding 'x' and 'res' really does move.  Entering from
// main() directly would reuse one frame at one address, and a replay reading
// 'x' through the recorded (stale) shareds pointer would still find the live
// value by accident.

#include <cstdio>

struct Bias3 {
  int p, q, r;
};

static Bias3 SavedBias = {1, 2, 3}; // p+q+r == 6

static volatile int StackSink = 0;

__attribute__((noinline)) static void clobber_stack(int base) {
  volatile int scratch[4096];

  for (int i = 0; i < 4096; ++i)
    scratch[i] = base + i;

  StackSink += scratch[base & 63];
}

__attribute__((noinline)) static int run_default_none_taskloop(int seed) {
  int x = seed;
  int res = 0;

#pragma omp taskgraph graph_id(9102)
  {
#pragma omp taskloop replayable num_tasks(8) default(none) shared(x)           \
    firstprivate(saved : SavedBias) reduction(+ : res)
    for (int i = 0; i < 16; ++i)
      res += x + i + SavedBias.p + SavedBias.q + SavedBias.r;
  }

  return res;
}

__attribute__((noinline)) static int call_with_depth(int seed, int depth) {
  volatile int padding[128];

  for (int i = 0; i < 128; ++i)
    padding[i] = seed + depth + i;

  StackSink += padding[(seed + depth) & 127];

  if (depth == 0)
    return run_default_none_taskloop(seed);
  return call_with_depth(seed, depth - 1);
}

int main() {
  bool failed = false;

  // sum_{i=0}^{15} (x + i + 6) = 16*x + 120 + 96 = 16*x + 216.
  const int first = call_with_depth(1, 0); // 16 + 216
  if (first != 232) {
    std::fprintf(stderr,
                 "FAIL default(none) taskloop record res=%d expected=232\n",
                 first);
    failed = true;
  }

  SavedBias = Bias3{100, 100, 100};

  clobber_stack(100 * 1000);
  const int second = call_with_depth(100, 3); // 1600 + 216
  clobber_stack(5 * 1000);
  const int third = call_with_depth(5, 6); // 80 + 216
  if (second != 1816 || third != 296) {
    std::fprintf(stderr,
                 "FAIL default(none) taskloop replay second=%d third=%d "
                 "expected=1816/296\n",
                 second, third);
    failed = true;
  }

  if (failed)
    return 1;

  std::fprintf(stderr,
               "PASS default(none) taskloop first=%d second=%d third=%d\n",
               first, second, third);
  return 0;
}

// CHECK: PASS default(none) taskloop first=232 second=1816 third=296
