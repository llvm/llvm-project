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

__attribute__((noinline)) static int run_taskgraph_nontrivial(int seed) {
  Tracker Obj(seed);
  int res = 0;

#pragma omp taskgraph graph_id(614)
  {
#pragma omp taskloop replayable num_tasks(8) shared(Obj) reduction(+ : res)
    for (int i = 0; i < 16; ++i) {
      res += Obj.Value + i;
    }
  }

  return res;
}

int main() {
  const int first = run_taskgraph_nontrivial(1);
  const int second = run_taskgraph_nontrivial(100);

  if (first != 136 || second != 1720 || Tracker::Ctors < 2 ||
      Tracker::Dtors < 2 || Tracker::Ctors != Tracker::Dtors) {
    std::fprintf(stderr,
                 "FAIL lexical nontrivial taskloop replay first=%d second=%d "
                 "ctors=%d dtors=%d\n",
                 first, second, Tracker::Ctors, Tracker::Dtors);
    return 1;
  }

  std::fprintf(stderr,
               "PASS lexical nontrivial taskloop replay first=%d second=%d "
               "ctors=%d dtors=%d\n",
               first, second, Tracker::Ctors, Tracker::Dtors);
  return 0;
}

// CHECK: PASS lexical nontrivial taskloop replay first=136 second=1720
