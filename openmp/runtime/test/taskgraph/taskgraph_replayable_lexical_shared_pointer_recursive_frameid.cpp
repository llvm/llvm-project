// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <cstdint>
#include <cstdio>

__attribute__((noinline)) static int expected_recursive(int depth, int seed,
                                                        int run_tag) {
  int value = seed;
  value += (depth + 1) * 3 + run_tag;
  if (depth == 0)
    return value;
  return value + expected_recursive(depth - 1, seed + 7, run_tag);
}

__attribute__((noinline)) static int run_recursive_frameid(int depth, int seed,
                                                           int run_tag) {
  int value = seed;
  int *ptr = &value;
  int *&ptr_ref = ptr;
  int out = -1;

  // Typically, if captured pointers refer to locations on the stack, that
  // would not be safe for taskgraph record/replay because we in general we
  // cannot rewrite such pointers to point to the current (live) stack frame.
  //
  // This is one possible way around that though: we keep a taskgraph record
  // per stack-depth, each of which may refer to the local stack frame.
  //
  // I probably wouldn't recommend use of this technique in production code.
  uintptr_t frame_gid = reinterpret_cast<uintptr_t>(__builtin_frame_address(0));

#pragma omp taskgraph graph_id(frame_gid)
  {
#pragma omp task shared(ptr_ref, out, depth, run_tag) depend(inout : value)
    {
      *ptr_ref += (depth + 1) * 3 + run_tag;
      out = *ptr_ref;
    }
  }

  if (depth == 0)
    return out;
  return out + run_recursive_frameid(depth - 1, seed + 7, run_tag);
}

int main() {
  const int depth = 3;
  int actual_sum = 0;
  int expected_sum = 0;

  for (int run = 0; run < 3; ++run) {
    int seed = 100 * run + 1;
    int actual = run_recursive_frameid(depth, seed, run);
    int expected = expected_recursive(depth, seed, run);
    if (actual != expected) {
      std::fprintf(stderr,
                   "FAIL recursive pointer frameid run=%d actual=%d expected=%d\n",
                   run, actual, expected);
      return 1;
    }
    actual_sum += actual;
    expected_sum += expected;
  }

  std::fprintf(stderr,
               "PASS recursive pointer frameid runs=3 total=%d expected=%d\n",
               actual_sum, expected_sum);
  return 0;
}

// CHECK: PASS recursive pointer frameid runs=3 total=
