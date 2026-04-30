// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s

// REQUIRES: omp_taskgraph_experimental

#include <atomic>
#include <cstdio>

// This builds a non-series-parallel dependency shape:
// A->B, A->C, B->D, C->D, C->E.

int main() {
  int deps[4] = {0, 0, 0, 0};

  for (int iter = 0; iter < 1000; ++iter) {
    std::atomic<int> sum{0};

#pragma omp parallel num_threads(4)
    {
#pragma omp single
      {
#pragma omp taskgraph graph_id(123) graph_reset(1)
        {
#pragma omp task depend(out : deps[0], deps[1])
          { sum.fetch_add(1, std::memory_order_relaxed); } // A
#pragma omp task depend(inout : deps[0])
          { sum.fetch_add(4, std::memory_order_relaxed); } // B
#pragma omp task depend(inout : deps[1])
          { sum.fetch_add(8, std::memory_order_relaxed); } // C
#pragma omp task depend(in : deps[0], deps[1], deps[2], deps[3])
          { sum.fetch_add(64, std::memory_order_relaxed); } // D
#pragma omp task depend(in : deps[1], deps[2])
          { sum.fetch_add(128, std::memory_order_relaxed); } // E
        }
      }
    }

    const int actual = sum.load(std::memory_order_relaxed);
    if (actual != 205) {
      std::fprintf(stderr, "FAIL iter=%d expected=205 actual=%d\n", iter,
                   actual);
      return 1;
    }
  }

  std::fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK: PASS
