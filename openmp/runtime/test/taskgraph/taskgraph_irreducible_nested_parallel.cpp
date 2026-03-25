// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A knot (irreducible region) one of whose members is a collapsed PARALLEL
// container that other members of the same knot depend on.
//
//        T1          T2
//       /  \        /  \
//      T4   \      /   T5        par = { X | Y }
//       |    [ par ]    |
//       |    /     \    |
//       T7 -'       '-- T8
//
// X and Y have identical predecessor and successor sets, so the twin-merge pass
// folds them into one PARALLEL region before the knot is carved; X and Y then
// feed T7 and T8, so the PARALLEL appears in their 'predecessors' lists.
//
// A PARALLEL does get an exec descriptor of its own, but it is the container's
// *gather* (entry) descriptor, which fires when the container starts.  An edge
// naming the PARALLEL must therefore resolve to its exit set -- both X and Y,
// since they run concurrently -- and not to that entry descriptor, or T7 and T8
// are released too early.  X and Y spin briefly to make an early release
// observable rather than merely possible.

#include <cstdio>

#define ITERS 200
#define SPIN 200000

static volatile int a, b, x, y, d, e;
static volatile int sink;
static int errors;

static void spin() {
  for (int i = 0; i < SPIN; ++i)
    sink = sink + 1;
}

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
#pragma omp task depend(in : a, b) depend(out : x)
        {
          spin();
          x = a + b;
        }
#pragma omp task depend(in : a, b) depend(out : y)
        {
          spin();
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

  std::printf("irreducible nested parallel errors: %d\n", errors);
  return errors != 0;
}

// CHECK: irreducible nested parallel errors: 0
