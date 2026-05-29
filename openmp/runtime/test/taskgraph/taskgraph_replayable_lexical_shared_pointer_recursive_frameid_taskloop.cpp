// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// XFAIL: *
// clang-format on

#include <cstdio>
#include <cstdint>

__attribute__((noinline)) static int expected_recursive(int depth, int seed,
                                                        int run_tag) {
  int value = 16 * (seed + (depth + 1) * 3 + run_tag) + 120;
  if (depth == 0)
    return value;
  return value + expected_recursive(depth - 1, seed + 7, run_tag);
}

__attribute__((noinline)) static int run_recursive_frameid(int depth, int seed,
                                                           int run_tag) {
  int value = seed;
  int *ptr = &value;
  int *&ptr_ref = ptr;
  int sum_delta = 0;
  uintptr_t frame_gid = reinterpret_cast<uintptr_t>(__builtin_frame_address(0));

  // Typically, if captured pointers refer to locations on the stack, that
  // would not be safe for taskgraph record/replay because we in general we
  // cannot rewrite such pointers to point to the current (live) stack frame.
  //
  // This is one possible way around that though: we keep a taskgraph record
  // per stack-depth, each of which may refer to the local stack frame.
  //
  // I probably wouldn't recommend use of this technique in production code.
#pragma omp taskgraph graph_id(frame_gid)
  {
#pragma omp taskloop replayable num_tasks(8) shared(ptr_ref, depth, run_tag) reduction(+ : sum_delta)
    for (int i = 0; i < 16; ++i) {
      int delta = (depth + 1) * 3 + run_tag + i;
      __atomic_fetch_add(ptr_ref, delta, __ATOMIC_RELAXED);
      sum_delta += delta;
    }
  }

  int local = value * 17 + sum_delta;

  if (depth == 0)
    return local;
  return local + run_recursive_frameid(depth - 1, seed + 7, run_tag);
}

int main() {
  const int depth = 3;
  int recorded_sum = 0;
  int replayed_sum = 0;

  for (int run = 0; run < 3; ++run) {
    int seed = 100 * run + 1;
    int val = run_recursive_frameid(depth, seed, run);
    if (run == 0)
      recorded_sum = val;
    else
      replayed_sum += val;
  }

  // With missing relocation for taskloop replay, recursive invocations that
  // mutate through shared-block pointers are expected to diverge from the
  // expected replay behavior.
  const int expected_replayed = 2 * recorded_sum;
  if (replayed_sum == expected_replayed) {
    std::fprintf(stderr,
                 "UNEXPECTED SUCCESS recursive pointer taskloop replay recorded=%d replayed_total=%d expected_total=%d\n",
                 recorded_sum, replayed_sum, expected_replayed);
    return 0;
  }

  std::fprintf(stderr,
               "EXPECTED FAILURE recursive pointer taskloop replay recorded=%d replayed_total=%d expected_total=%d\n",
               recorded_sum, replayed_sum, expected_replayed);
  return 1;
}

// CHECK: EXPECTED FAILURE recursive pointer taskloop replay
