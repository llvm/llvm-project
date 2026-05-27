// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <cstdio>

__attribute__((noinline)) static int run_taskgraph_pointer_shared(int seed) {
  int value = seed;
  int *ptr = &value;
  int out = -1;

#pragma omp taskgraph graph_id(402)
  {
#pragma omp task replayable(1) shared(ptr, out) depend(inout : value)
    {
      *ptr += 3;
      out = *ptr;
    }
  }

  return out;
}

int main() {
  const int first = run_taskgraph_pointer_shared(1);
  const int second = run_taskgraph_pointer_shared(100);

  if (first != 4 || second != 103) {
    std::fprintf(stderr,
                 "FAIL lexical pointer shared replay first=%d second=%d "
                 "expected=4/103\n",
                 first, second);
    return 1;
  }

  std::fprintf(stderr,
               "PASS lexical pointer shared replay first=%d second=%d\n", first,
               second);
  return 0;
}

// CHECK: PASS lexical pointer shared replay first=4 second=103
