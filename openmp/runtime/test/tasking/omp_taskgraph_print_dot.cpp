// REQUIRES: omp_taskgraph_experimental
// RUN: %libomp-cxx-compile-and-run
// RUN: cat tdg_17353.dot | FileCheck %s
// RUN: rm -f tdg_17353.dot

#include <cstdlib>
#include <cassert>

// Compiler-generated code (emulation)
typedef struct ident {
  void *dummy;
} ident_t;

void func(int *num_exec) {
#pragma omp atomic
  (*num_exec)++;
}

int main() {
  int num_exec = 0;
  int x, y;

  setenv("KMP_TDG_DOT", "TRUE", 1);

#pragma omp parallel
#pragma omp single
  {
#pragma omp taskgraph
    {
#pragma omp task depend(out : x)
      func(&num_exec);
#pragma omp task depend(in : x) depend(out : y)
      func(&num_exec);
#pragma omp task depend(in : y)
      func(&num_exec);
#pragma omp task depend(in : y)
      func(&num_exec);
    }
  }

  assert(num_exec == 4);

  return 0;
}

// CHECK:      digraph TDG {
// CHECK-NEXT:    compound=true
// CHECK-NEXT:    subgraph cluster {
// CHECK-NEXT:       label=TDG_17353
// CHECK-NEXT:       0[style=bold]
// CHECK-NEXT:       1[style=bold]
// CHECK-NEXT:       2[style=bold]
// CHECK-NEXT:       3[style=bold]
// CHECK-NEXT:    }
// CHECK-NEXT:    0 -> 1
// CHECK-NEXT:    1 -> 2
// CHECK-NEXT:    1 -> 3
// CHECK-NEXT: }
