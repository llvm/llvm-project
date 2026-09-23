// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A recording that ends up with no task nodes is left with a null region tree
// by __kmp_build_taskgraph, so tearing such a record down (which is what
// graph_reset does before re-recording) must tolerate that.

#include <atomic>
#include <cstdio>

int main() {
  constexpr int NumIters = 4;

  std::atomic<int> total{0};

#pragma omp parallel num_threads(4)
  {
#pragma omp single
    {
      // An empty body: nothing is recorded, so the record has no regions.
      for (int iter = 0; iter < NumIters; ++iter) {
#pragma omp taskgraph graph_id(1) graph_reset(1)
        {
        }
      }

      // Same, but alternating empty and non-empty bodies, so a re-record has
      // to cope with the previous record being empty and vice versa.
      for (int iter = 0; iter < NumIters; ++iter) {
#pragma omp taskgraph graph_id(2) graph_reset(1)
        {
          if (iter & 1) {
            for (int i = 0; i < 4; ++i) {
#pragma omp task
              total.fetch_add(1, std::memory_order_relaxed);
            }
          }
        }
      }
    }
  }

  const int Expected = (NumIters / 2) * 4;
  const int Actual = total.load(std::memory_order_relaxed);

  if (Actual != Expected) {
    std::fprintf(stderr, "FAIL empty graph_reset total=%d expected=%d\n",
                 Actual, Expected);
    return 1;
  }

  std::fprintf(stderr, "PASS empty graph_reset total=%d\n", Actual);
  return 0;
}

// CHECK: PASS empty graph_reset total=
