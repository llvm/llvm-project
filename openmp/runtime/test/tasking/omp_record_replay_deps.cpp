// REQUIRES: ompx_taskgraph
// RUN: %libomp-cxx-compile-and-run
#include <iostream>
#include <cassert>
#define NT 100
#define MULTIPLIER 100
#define DECREMENT 5

int val;
// Compiler-generated code (emulation)
typedef struct ident {
    void* dummy;
} ident_t;


#ifdef __cplusplus
extern "C" {
  int __kmpc_global_thread_num(ident_t *);
  int __kmpc_start_record_task(ident_t *, int, int, int);
  void __kmpc_end_record_task(ident_t *, int, int, int);
}
#endif

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
  val = 0;
  int *x, *y;
  #pragma omp parallel
  #pragma omp single
  for (int iter = 0; iter < NT; ++iter) {
    int gtid = __kmpc_global_thread_num(nullptr);
    int res =  __kmpc_start_record_task(nullptr, gtid, /* kmp_tdg_flags */0, /* tdg_id */0);
    if (res) {
      #pragma omp task depend(out:y)
      add();
      #pragma omp task depend(out:x)
      sub();
      #pragma omp task depend(in:x,y)
      mult();
    }
    __kmpc_end_record_task(nullptr, gtid, /* kmp_tdg_flags */0, /* tdg_id */0);
  }
  assert(val==0);

  std::cout << "Passed" << std::endl;
  return 0;
}
// CHECK: Passed
