// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// default(shared) on a replayable taskloop: 'x' is implicitly shared and
// relocated across replays; 'res' is an explicit reduction (which overrides the
// default) and the loop variable is predetermined private.
//
// Both 'x' and 'res' live in run_default_shared_taskloop's frame, and the
// taskgraph is entered through call_with_depth() at a different recursion depth
// per encounter (with clobber_stack() overwriting the abandoned frames in
// between) so that frame genuinely moves.  Entering from main() directly would
// reuse one frame at one address, letting a replay that read 'x' through the
// recorded (stale) shareds pointer land on the live object by accident, so the
// relocation claim above would not be tested.

#include <cstdio>

static volatile int StackSink = 0;

__attribute__((noinline)) static void clobber_stack(int base) {
  volatile int scratch[4096];

  for (int i = 0; i < 4096; ++i)
    scratch[i] = base + i;

  StackSink += scratch[base & 63];
}

__attribute__((noinline)) static int run_default_shared_taskloop(int seed) {
  int x = seed;
  int res = 0;

#pragma omp taskgraph graph_id(9104)
  {
#pragma omp taskloop replayable num_tasks(8) default(shared) reduction(+ : res)
    for (int i = 0; i < 16; ++i)
      res += x + i;
  }

  return res;
}

__attribute__((noinline)) static int call_with_depth(int seed, int depth) {
  volatile int padding[128];

  for (int i = 0; i < 128; ++i)
    padding[i] = seed + depth + i;

  StackSink += padding[(seed + depth) & 127];

  if (depth == 0)
    return run_default_shared_taskloop(seed);
  return call_with_depth(seed, depth - 1);
}

int main() {
  bool failed = false;

  // sum_{i=0}^{15} (x + i) = 16*x + 120.
  const int first = call_with_depth(1, 0); // 16 + 120
  clobber_stack(100 * 1000);
  const int second = call_with_depth(100, 3); // 1600 + 120
  clobber_stack(5 * 1000);
  const int third = call_with_depth(5, 6); // 80 + 120
  if (first != 136 || second != 1720 || third != 200) {
    std::fprintf(stderr,
                 "FAIL default(shared) taskloop first=%d second=%d third=%d "
                 "expected=136/1720/200\n",
                 first, second, third);
    failed = true;
  }

  if (failed)
    return 1;

  std::fprintf(stderr,
               "PASS default(shared) taskloop first=%d second=%d third=%d\n",
               first, second, third);
  return 0;
}

// CHECK: PASS default(shared) taskloop first=136 second=1720 third=200
