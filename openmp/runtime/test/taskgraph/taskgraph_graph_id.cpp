// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <atomic>
#include <cstdio>

int main() {
  constexpr int NumIters = 12;
  constexpr int WorkA = 16;
  constexpr int WorkB = 24;

  std::atomic<int> total{0};

#pragma omp parallel num_threads(4)
  {
#pragma omp single
    {
      for (int iter = 0; iter < NumIters; ++iter) {
        const int gid = iter & 1;

#pragma omp taskgraph graph_id(gid)
        {
          if (gid == 0) {
            for (int i = 0; i < WorkA; ++i) {
#pragma omp task
              total.fetch_add(1, std::memory_order_relaxed);
            }
          } else {
            for (int i = 0; i < WorkB; ++i) {
#pragma omp task
              total.fetch_add(1, std::memory_order_relaxed);
            }
          }
        }
      }
    }
  }

  const int NumEven = (NumIters + 1) / 2;
  const int NumOdd = NumIters / 2;
  const int Expected = NumEven * WorkA + NumOdd * WorkB;
  const int Actual = total.load(std::memory_order_relaxed);

  if (Actual != Expected) {
    std::fprintf(stderr, "FAIL graph_id total=%d expected=%d\n", Actual,
                 Expected);
    return 1;
  }

  std::fprintf(stderr, "PASS graph_id total=%d\n", Actual);
  return 0;
}

// CHECK: PASS graph_id total=
