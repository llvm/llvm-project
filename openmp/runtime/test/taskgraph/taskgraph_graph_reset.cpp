// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <atomic>
#include <cstdio>

int main() {
  constexpr int NumIters = 10;
  constexpr int GraphId = 7;

  std::atomic<int> total{0};

#pragma omp parallel num_threads(4)
  {
#pragma omp single
    {
      for (int iter = 0; iter < NumIters; ++iter) {
        const bool odd = (iter & 1) != 0;

        // Alternate the taskgraph shape every iteration. graph_reset(1)
        // requires the runtime to re-record rather than replay stale shape.
#pragma omp taskgraph graph_id(GraphId) graph_reset(1)
        {
          if (odd) {
            for (int i = 0; i < 40; ++i) {
#pragma omp task
              total.fetch_add(2, std::memory_order_relaxed);
            }
          } else {
            for (int i = 0; i < 10; ++i) {
#pragma omp task
              total.fetch_add(1, std::memory_order_relaxed);
            }
          }
        }
      }
    }
  }

  const int NumOdd = NumIters / 2;
  const int NumEven = NumIters - NumOdd;
  const int Expected = NumOdd * 40 * 2 + NumEven * 10;
  const int Actual = total.load(std::memory_order_relaxed);

  if (Actual != Expected) {
    std::fprintf(stderr, "FAIL graph_reset total=%d expected=%d\n", Actual,
                 Expected);
    return 1;
  }

  std::fprintf(stderr, "PASS graph_reset total=%d\n", Actual);
  return 0;
}

// CHECK: PASS graph_reset total=
