// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <cstdio>

struct Tracker {
  static int Ctors;
  static int Dtors;

  int Value;

  explicit Tracker(int V) : Value(V) { ++Ctors; }
  ~Tracker() { ++Dtors; }

  void bump(int Delta) { Value += Delta; }
};

int Tracker::Ctors = 0;
int Tracker::Dtors = 0;

__attribute__((noinline)) static int expected_recursive(int depth, int seed,
                                                        int run_tag) {
  int local = 16 * (seed + (depth + 1) * 5 + run_tag) + 120;
  if (depth == 0)
    return local;
  return local + expected_recursive(depth - 1, seed + 9, run_tag);
}

__attribute__((noinline)) static int run_recursive_nontrivial(int depth, int seed,
                                                              int run_tag) {
  Tracker Obj(seed);
  int res = 0;

  int gid = 620 + depth;
#pragma omp taskgraph graph_id(gid)
  {
#pragma omp taskloop replayable num_tasks(8) shared(Obj, depth, run_tag) reduction(+ : res)
    for (int i = 0; i < 16; ++i) {
      res += Obj.Value + (depth + 1) * 5 + run_tag + i;
    }
  }

  if (depth == 0)
    return res;
  return res + run_recursive_nontrivial(depth - 1, seed + 9, run_tag);
}

int main() {
  const int depth = 3;
  int total_actual = 0;
  int total_expected = 0;

  for (int run = 0; run < 3; ++run) {
    const int seed = 100 * run + 1;
    const int actual = run_recursive_nontrivial(depth, seed, run);
    const int expected = expected_recursive(depth, seed, run);

    if (actual != expected) {
      std::fprintf(stderr,
                   "FAIL recursive nontrivial taskloop run=%d actual=%d expected=%d\n",
                   run, actual, expected);
      return 1;
    }

    total_actual += actual;
    total_expected += expected;
  }

  if (Tracker::Ctors != Tracker::Dtors || Tracker::Ctors < 12) {
    std::fprintf(stderr,
                 "FAIL recursive nontrivial taskloop lifetime ctors=%d dtors=%d total=%d expected=%d\n",
                 Tracker::Ctors, Tracker::Dtors, total_actual, total_expected);
    return 1;
  }

  std::fprintf(stderr,
               "PASS recursive nontrivial taskloop total=%d expected=%d ctors=%d dtors=%d\n",
               total_actual, total_expected, Tracker::Ctors, Tracker::Dtors);
  return 0;
}

// CHECK: PASS recursive nontrivial taskloop total=
