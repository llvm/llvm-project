// clang-format off
// REQUIRES: omp_taskgraph_experimental
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run
// clang-format on
//
// NxN lattice "wavefront": cell (i,j) depends on (i-1,j) and (i,j-1).  This is
// the canonical non-series-parallel (irreducible) dependence graph.  It used to
// livelock the taskgraph region builder, which tried to make the graph
// reducible by node-splitting -- a rewrite that does not converge on
// lattice-like graphs.  The builder now carves the residual into a single
// TASKGRAPH_REGION_IRREDUCIBLE region instead, which is guaranteed to
// terminate.
//
// Each cell accumulates the number of monotone lattice paths reaching it
// (val[i][j] == val[i-1][j] + val[i][j-1], with val[0][0] == 1), i.e. the
// binomial C(i+j, i).  N is kept small enough that the values fit in long long.
#include <cassert>
#include <iostream>

#define N 12
#define ITERS 20

static long long val[N][N];
static long long ref[N][N];

int main() {
  // Serial reference, computed with the same recurrence to avoid any
  // closed-form overflow surprises.
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      ref[i][j] =
          (i == 0 && j == 0)
              ? 1
              : ((i > 0 ? ref[i - 1][j] : 0) + (j > 0 ? ref[i][j - 1] : 0));

  for (int iter = 0; iter < ITERS; ++iter) {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        val[i][j] = -1;

#pragma omp parallel
#pragma omp single
    {
#pragma omp taskgraph
      {
        for (int i = 0; i < N; ++i)
          for (int j = 0; j < N; ++j) {
            long long *me = &val[i][j];
            long long *up = (i > 0) ? &val[i - 1][j] : nullptr;
            long long *left = (j > 0) ? &val[i][j - 1] : nullptr;
            if (i == 0 && j == 0) {
#pragma omp task depend(out : me[0])
              {
                *me = 1;
              }
            } else if (i == 0) {
#pragma omp task depend(in : left[0]) depend(out : me[0])
              {
                *me = *left;
              }
            } else if (j == 0) {
#pragma omp task depend(in : up[0]) depend(out : me[0])
              {
                *me = *up;
              }
            } else {
#pragma omp task depend(in : up[0], left[0]) depend(out : me[0])
              {
                *me = *up + *left;
              }
            }
          }
      }
    }

    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        assert(val[i][j] == ref[i][j] && "wavefront value mismatch");
  }

  std::cout << "Passed" << std::endl;
  return 0;
}
// CHECK: Passed
