// REQUIRES: omp_taskgraph_experimental
// RUN: %libomp-cxx-compile-and-run
#include <iostream>
#include <cassert>
#define NT 20
#define MULTIPLIER 100
#define DECREMENT 5

// Compiler-generated code (emulation)
typedef struct ident {
  void *dummy;
} ident_t;

int val;

void sub() {
#pragma omp atomic
  val -= DECREMENT;
}

void add() {
#pragma omp atomic
  val += DECREMENT;
}

void mult() {
  // no atomicity needed, can only be executed by 1 thread
  // and no concurrency with other tasks possible
  val *= MULTIPLIER;
}

int main() {
  int num_tasks = 0;
  int *x, *y;
#pragma omp parallel
#pragma omp single
  for (int iter = 0; iter < NT; ++iter) {
#pragma omp taskgraph
    {
      num_tasks++;
#pragma omp task depend(out : y)
      add();
#pragma omp task depend(out : x)
      sub();
#pragma omp task depend(in : x, y)
      mult();
    }
#pragma omp taskgraph
    {
      num_tasks++;
#pragma omp task depend(out : y)
      add();
#pragma omp task depend(out : x)
      sub();
#pragma omp task depend(in : x, y)
      mult();
    }
  }

  assert(num_tasks == 2);
  assert(val == 0);

  std::cout << "Passed" << std::endl;
  return 0;
}
// CHECK: Passed
