// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental, linux
// clang-format on

// Every graph_reset tears the previous record down and records afresh, so the
// teardown has to return everything the recording took.  It did not: the
// per-node region array is a single allocation whose elements after the first
// are not on the chain either cleanup walk follows, so their edge lists were
// never released, and the transient depnodes the build borrows to discover the
// edges kept their successor lists.  That leaked about a kilobyte per recorded
// task per reset -- unbounded in any program that re-records in a loop.
//
// Measured as resident-set growth, because what is leaked is small blocks that
// libomp's own allocator would otherwise recycle: the bug is invisible to a
// malloc-level leak checker (the blocks are freed by the process's own
// teardown) and only shows up as memory the running program never gets back.
//
// The graphs below cover the shapes that reach the different cleanup paths: a
// wide fan with chained dependencies for the ordinary series/parallel collapse,
// and a non-series-parallel knot for the irreducible carve, which is the one
// case that deliberately keeps edge lists alive until teardown.
//
// Dependences are arranged so that every address read in an iteration is also
// written in it.  A read of an address that is never written is remembered by
// the runtime until some task writes it, which grows without bound whether or
// not a taskgraph is involved, and would swamp what this test measures.

#include <atomic>
#include <cstdio>
#include <unistd.h>

static long rss_kb() {
  std::FILE *f = std::fopen("/proc/self/statm", "r");
  if (!f)
    return -1;
  long pages_total = 0, pages_resident = 0;
  int got = std::fscanf(f, "%ld %ld", &pages_total, &pages_resident);
  std::fclose(f);
  if (got != 2)
    return -1;
  return pages_resident * (long)(sysconf(_SC_PAGESIZE) / 1024);
}

static std::atomic<int> Counter{0};

template <typename BodyTy> static void replay(int iters, BodyTy &&body) {
  for (int iter = 0; iter < iters; ++iter)
    body();
}

int main() {
  // The pre-fix leak is ~1 kB per recorded task per reset, so the fan graph
  // alone leaked ~80 MB over the measured loop.  Anything under a few MB of
  // growth means the teardown is returning what the recording took; the slack
  // is for the allocator's own arena granularity, not for a per-reset leak.
  constexpr int WarmupIters = 200;
  constexpr int MeasuredIters = 4000;
  constexpr long LimitKB = 8 * 1024;

  int fan[16] = {0};
  int knot[2] = {0};
  long before = 0, after = 0;

#pragma omp parallel num_threads(4)
  {
#pragma omp single
    {
      auto fan_graph = [&]() {
#pragma omp taskgraph graph_id(1) graph_reset(1)
        {
          for (int i = 0; i < 16; ++i) {
#pragma omp task depend(inout : fan[i])
            Counter.fetch_add(1, std::memory_order_relaxed);
          }
        }
      };

      // A -> {B, C}, {B, C} -> D, C -> E: not series-parallel, so the build
      // carves it into an IRREDUCIBLE region whose children keep their edges.
      auto knot_graph = [&]() {
#pragma omp taskgraph graph_id(2) graph_reset(1)
        {
#pragma omp task depend(out : knot[0], knot[1])
          Counter.fetch_add(1, std::memory_order_relaxed);
#pragma omp task depend(inout : knot[0])
          Counter.fetch_add(1, std::memory_order_relaxed);
#pragma omp task depend(inout : knot[1])
          Counter.fetch_add(1, std::memory_order_relaxed);
#pragma omp task depend(in : knot[0], knot[1])
          Counter.fetch_add(1, std::memory_order_relaxed);
#pragma omp task depend(in : knot[1])
          Counter.fetch_add(1, std::memory_order_relaxed);
        }
      };

      replay(WarmupIters, fan_graph);
      replay(WarmupIters, knot_graph);
      before = rss_kb();

      replay(MeasuredIters, fan_graph);
      replay(MeasuredIters, knot_graph);
      after = rss_kb();
    }
  }

  if (before < 0 || after < 0) {
    std::fprintf(stderr, "FAIL could not read resident set size\n");
    return 1;
  }

  const long growth = after - before;
  if (growth > LimitKB) {
    std::fprintf(stderr,
                 "FAIL taskgraph reset leaks: resident set grew %ld kB over "
                 "%d resets (limit %ld kB)\n",
                 growth, 2 * MeasuredIters, LimitKB);
    return 1;
  }

  std::fprintf(stderr, "PASS taskgraph reset growth %ld kB over %d resets\n",
               growth, 2 * MeasuredIters);
  return 0;
}

// CHECK: PASS taskgraph reset growth
