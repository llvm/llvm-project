// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// The same shape as taskgraph_irreducible_nested_sequential.cpp reached through
// the other container type that carries no exec descriptor of its own.
//
//        T1          T2
//       /  \        /  \
//      T4   \      /   T5        excl = { X | Y }  (mutexinoutset)
//       |   [ excl ]    |
//       |    /     \    |
//       T7 -'       '-- T8
//
// X and Y have identical predecessor and successor sets, so the twin-merge pass
// folds them into one region before the knot is carved; because they carry a
// mutexinoutset, the merge produces an EXCLUSIVE rather than a PARALLEL.  Like
// SEQUENTIAL, EXCLUSIVE is expanded by recursing into its children and is never
// given a descriptor, so an edge naming it has to resolve to the exit descrs of
// its last child.
//
// Being the only shape that puts a mutex set inside a knot, this also covers
// __kmp_taskgraph_gather_mutex_sets descending into an IRREDUCIBLE: without
// that, the EXCLUSIVE keeps a null mutexset and
// __kmp_taskgraph_find_exclusive_regions dereferences it.

#include <cstdio>

#define ITERS 200

static volatile int a, b, x, y, d, e;
static volatile int m;
static int errors;

int main() {
  for (int iter = 0; iter < ITERS; ++iter) {
    a = b = x = y = d = e = 0;
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
#pragma omp task depend(in : a, b) depend(mutexinoutset : m) depend(out : x)
        {
          m = m + 1;
          x = a + b;
        }
#pragma omp task depend(in : a, b) depend(mutexinoutset : m) depend(out : y)
        {
          m = m + 1;
          y = a + b + 1;
        }
#pragma omp task depend(in : a) depend(out : d)
        { d = a + 10; }
#pragma omp task depend(in : b) depend(out : e)
        { e = b + 20; }
        // clang-format on
#pragma omp task depend(in : x, y, d)
        {
          if (x != 3 || y != 4 || d != 11)
            ++errors;
        }
#pragma omp task depend(in : x, y, e)
        {
          if (x != 3 || y != 4 || e != 22)
            ++errors;
        }
      }
    }
  }

  std::printf("irreducible nested exclusive errors: %d\n", errors);
  return errors != 0;
}

// CHECK: irreducible nested exclusive errors: 0
