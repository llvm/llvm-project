// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A knot (irreducible region) one of whose members is a collapsed SEQUENTIAL
// container that other members of the same knot depend on.
//
//        T1          T2
//       /  \        /  \
//      T4   \      /   T5        seq = (T3 ; T6)
//       |    [ seq ]    |
//       |    /     \    |
//       T7 -'       '-- T8
//
// T3 -> T6 is a clean 1:1 producer/consumer chain, so the series collapse folds
// it into a SEQUENTIAL region before the residual knot is carved; T6 then feeds
// T7 and T8, so the SEQUENTIAL appears in their 'predecessors' lists.
//
// A SEQUENTIAL (like an EXCLUSIVE) gets no exec descriptor of its own, so an
// edge naming it has to resolve to the exit descrs of its last child.  Note the
// carved members are in reverse *pre*order, not topological order
// (__kmp_taskgraph_region_dfs numbers on entry), so the SEQUENTIAL is reached
// from T7 before it would have been built in member order.

#include <cstdio>

#define ITERS 200

static volatile int a, b, c, d, e, f;
static int errors;

int main() {
  for (int iter = 0; iter < ITERS; ++iter) {
    a = b = c = d = e = f = 0;
#pragma omp parallel
#pragma omp single
    {
#pragma omp taskgraph
      {
        // clang-format off
#pragma omp task depend(out : a)
        { a = 1; }
#pragma omp task depend(out : b)
        { b = 2; }
#pragma omp task depend(in : a, b) depend(out : c)
        { c = a + b; }
#pragma omp task depend(in : a) depend(out : d)
        { d = a + 10; }
#pragma omp task depend(in : b) depend(out : e)
        { e = b + 20; }
#pragma omp task depend(in : c) depend(out : f)
        { f = c + 100; }
        // clang-format on
#pragma omp task depend(in : d, f)
        {
          if (d != 11 || f != 103)
            ++errors;
        }
#pragma omp task depend(in : e, f)
        {
          if (e != 22 || f != 103)
            ++errors;
        }
      }
    }
    if (c != 3 || f != 103)
      ++errors;
  }

  std::printf("irreducible nested sequential errors: %d\n", errors);
  return errors != 0;
}

// CHECK: irreducible nested sequential errors: 0
