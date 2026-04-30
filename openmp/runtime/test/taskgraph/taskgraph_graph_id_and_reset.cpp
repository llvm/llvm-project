// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <atomic>
#include <cstdio>

// Exercise graph_id and graph_reset together on a single taskgraph construct.
//
// graph_id is per-taskgraph: the runtime keeps one record per (construct,
// graph_id) pair.  Here we feed the same construct two different graph_ids
// alternately (0, 1, 0, 1, ...) so two records coexist for the same
// directive.  graph_reset is then driven by a separate condition: every
// fourth visit, we ask the runtime to re-record the graph_id we are about
// to use, forcing it through the expiry-and-re-record path while the other
// graph_id's record stays intact and continues to be replayed.

int main() {
  constexpr int NumIters = 40;
  constexpr int TasksPerVisit = 6;

  std::atomic<int> total{0};

#pragma omp parallel num_threads(4)
  {
#pragma omp single
    {
      for (int iter = 0; iter < NumIters; ++iter) {
        const int gid = iter & 1; // alternate two records
        const bool reset = (iter % 4) == 3; // periodically force re-record

#pragma omp taskgraph graph_id(gid) graph_reset(reset)
        {
          for (int i = 0; i < TasksPerVisit; ++i) {
#pragma omp task
            total.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    }
  }

  const int Expected = NumIters * TasksPerVisit;
  const int Actual = total.load(std::memory_order_relaxed);

  if (Actual != Expected) {
    std::fprintf(stderr, "FAIL graph_id+reset total=%d expected=%d\n", Actual,
                 Expected);
    return 1;
  }

  std::fprintf(stderr, "PASS graph_id+reset total=%d\n", Actual);
  return 0;
}

// CHECK: PASS graph_id+reset total=
