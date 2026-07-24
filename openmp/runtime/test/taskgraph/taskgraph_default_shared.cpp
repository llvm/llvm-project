// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// default(shared) on a replayable task: 'x' and 'out' get an implicit shared
// attribute rather than being listed explicitly.  Because they are shared by
// reference and live on a stack frame that moves between the recording call and
// the replay calls, this specifically exercises shared-by-reference relocation
// under an implicit (default) data-sharing attribute.
//
// The frame only actually moves because the taskgraph is entered through
// call_with_depth() at a different recursion depth per encounter, with
// clobber_stack() overwriting the abandoned frames in between.  Calling
// run_default_shared() directly from main() would reuse one frame at one
// address on every encounter, so a replay reading and writing through the
// recorded (stale) shareds pointer would still hit the live objects by accident
// and relocation would not be exercised at all.

#include <cstdio>

static volatile int StackSink = 0;

__attribute__((noinline)) static void clobber_stack(int base) {
  volatile int scratch[4096];

  for (int i = 0; i < 4096; ++i)
    scratch[i] = base + i;

  StackSink += scratch[base & 63];
}

__attribute__((noinline)) static int run_default_shared(int seed) {
  int x = seed;
  int out = -1;

#pragma omp taskgraph graph_id(9103)
  {
#pragma omp task default(shared) depend(inout : x)
    {
      x += 5;
      out = x;
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
    return run_default_shared(seed);
  return call_with_depth(seed, depth - 1);
}

int main() {
  bool failed = false;

  const int first = call_with_depth(1, 0);
  clobber_stack(100 * 1000);
  const int second = call_with_depth(100, 3);
  clobber_stack(5 * 1000);
  const int third = call_with_depth(5, 6);
  if (first != 6 || second != 105 || third != 10) {
    std::fprintf(stderr,
                 "FAIL default(shared) task first=%d second=%d third=%d "
                 "expected=6/105/10\n",
                 first, second, third);
    failed = true;
  }

  if (failed)
    return 1;

  std::fprintf(stderr,
               "PASS default(shared) task first=%d second=%d third=%d\n", first,
               second, third);
  return 0;
}

// CHECK: PASS default(shared) task first=6 second=105 third=10
