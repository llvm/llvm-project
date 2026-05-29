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
  int local = seed + (depth + 1) * 5 + run_tag;
  if (depth == 0)
    return local;
  return local + expected_recursive(depth - 1, seed + 9, run_tag);
}

__attribute__((noinline)) static int run_recursive_nontrivial(int depth, int seed,
                                                              int run_tag) {
  Tracker Obj(seed);
  int out = -1;

  int gid = 500 + depth;
#pragma omp taskgraph graph_id(gid)
  {
#pragma omp task replayable(1) shared(Obj, out, depth, run_tag)
    {
      Obj.bump((depth + 1) * 5 + run_tag);
      out = Obj.Value;
    }
  }

  if (depth == 0)
    return out;
  return out + run_recursive_nontrivial(depth - 1, seed + 9, run_tag);
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
                   "FAIL recursive nontrivial run=%d actual=%d expected=%d\n",
                   run, actual, expected);
      return 1;
    }

    total_actual += actual;
    total_expected += expected;
  }

  if (Tracker::Ctors != Tracker::Dtors || Tracker::Ctors < 12) {
    std::fprintf(stderr,
                 "FAIL recursive nontrivial lifetime ctors=%d dtors=%d total=%d expected=%d\n",
                 Tracker::Ctors, Tracker::Dtors, total_actual, total_expected);
    return 1;
  }

  std::fprintf(stderr,
               "PASS recursive nontrivial total=%d expected=%d ctors=%d dtors=%d\n",
               total_actual, total_expected, Tracker::Ctors, Tracker::Dtors);
  return 0;
}

// CHECK: PASS recursive nontrivial total=
