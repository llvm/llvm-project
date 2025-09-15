// REQUIRES: omp_taskgraph_experimental
// RUN: %libomp-cxx-compile-and-run
#include <iostream>
#include <cassert>
#define NT 100

// Compiler-generated code (emulation)
typedef struct ident {
  void *dummy;
} ident_t;

void func(int *num_exec) { (*num_exec)++; }

int main() {
  int num_exec = 0;
  int num_tasks = 0;
  int x = 0;
#pragma omp parallel
#pragma omp single
  for (int iter = 0; iter < NT; ++iter) {
#pragma omp taskgraph
    {
      num_tasks++;
#pragma omp task
      func(&num_exec);
    }
  }

  assert(num_tasks == 1);
  assert(num_exec == NT);

  std::cout << "Passed" << std::endl;
  return 0;
}
// CHECK: Passed
