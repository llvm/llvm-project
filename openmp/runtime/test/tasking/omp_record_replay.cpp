// REQUIRES: ompx_taskgraph
// RUN: %libomp-cxx-compile-and-run
#include <iostream>
#include <cassert>
#define NT 100

// Compiler-generated code (emulation)
typedef struct ident {
    void* dummy;
} ident_t;


#ifdef __cplusplus
extern "C" {
  int __kmpc_global_thread_num(ident_t *);
  int __kmpc_start_record_task(ident_t *, int, int, int);
  void __kmpc_end_record_task(ident_t *, int, int , int);
}
#endif

void func(int *num_exec) {
  (*num_exec)++;
}

int main() {
  int num_exec = 0;
  int num_tasks = 0;
  int x=0;
  #pragma omp parallel
  #pragma omp single
  for (int iter = 0; iter < NT; ++iter) {
    int gtid = __kmpc_global_thread_num(nullptr);
    int res =  __kmpc_start_record_task(nullptr, gtid, /* kmp_tdg_flags */ 0, /* tdg_id */0);
    if (res) {
      num_tasks++;
      #pragma omp task 
      func(&num_exec);
    }
    __kmpc_end_record_task(nullptr, gtid, /* kmp_tdg_flags */0, /* tdg_id */0);
  }

  assert(num_tasks==1);
  assert(num_exec==NT);

  std::cout << "Passed" << std::endl;
  return 0;
}
// CHECK: Passed
