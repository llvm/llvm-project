// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <cstdio>
#include <cstdint>

__attribute__((noinline)) static int expected_recursive(int depth, int seed,
                                                        int run_tag) {
  // Mirror run_recursive_frameid: the taskloop adds, over i in [0, 16),
  // delta = (depth + 1) * 3 + run_tag + i both to 'value' (through the shared
  // pointer) and to the reduction 'sum_delta'.
  int sum_delta = 16 * ((depth + 1) * 3 + run_tag) + 120;
  int value = seed + sum_delta;
  int local = value * 17 + sum_delta;
  if (depth == 0)
    return local;
  return local + expected_recursive(depth - 1, seed + 7, run_tag);
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
#pragma omp taskloop replayable num_tasks(8) shared(ptr_ref, depth, run_tag)   \
    reduction(+ : sum_delta)
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
  int actual_sum = 0;
  int expected_sum = 0;
  int failures = 0;

#pragma omp parallel
  {
#pragma omp single
    {
      for (int run = 0; run < 3; ++run) {
        int seed = 100 * run + 1;
        int actual = run_recursive_frameid(depth, seed, run);
        int expected = expected_recursive(depth, seed, run);
        if (actual != expected) {
          std::fprintf(
              stderr,
              "FAIL recursive pointer taskloop replay run=%d actual=%d "
              "expected=%d\n",
              run, actual, expected);
          ++failures;
        }
        actual_sum += actual;
        expected_sum += expected;
      }
    }
  }

  if (failures != 0)
    return 1;

  std::fprintf(stderr,
               "PASS recursive pointer taskloop replay runs=3 total=%d "
               "expected=%d\n",
               actual_sum, expected_sum);
  return 0;
}

// CHECK: PASS recursive pointer taskloop replay
