// REQUIRES: ompx_taskgraph
// RUN: %libomp-cxx-compile-and-run
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

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
  #pragma omp atomic
  (*num_exec)++;
}

std::string tdg_string= "digraph TDG {\n"
"   compound=true\n"
"   subgraph cluster {\n"
"      label=TDG_0\n"
"      0[style=bold]\n"
"      1[style=bold]\n"
"      2[style=bold]\n"
"      3[style=bold]\n"
"   }\n"
"   0 -> 1 \n"
"   1 -> 2 \n"
"   1 -> 3 \n"
"}";

int main() {
  int num_exec = 0;
  int x, y;

  setenv("KMP_TDG_DOT","TRUE",1);
  remove("tdg_0.dot");

  #pragma omp parallel
  #pragma omp single
  {
    int gtid = __kmpc_global_thread_num(nullptr);
    int res = __kmpc_start_record_task(nullptr, gtid, /* kmp_tdg_flags */ 0, /* tdg_id */ 0);
    if (res) {
      #pragma omp task depend(out : x)
      func(&num_exec);
      #pragma omp task depend(in : x) depend(out : y)
      func(&num_exec);
      #pragma omp task depend(in : y)
      func(&num_exec);
      #pragma omp task depend(in : y)
      func(&num_exec);
    }

    __kmpc_end_record_task(nullptr, gtid, /* kmp_tdg_flags */ 0, /* tdg_id */ 0);
  }

  assert(num_exec == 4);

  std::ifstream tdg_file("tdg_0.dot");
  assert(tdg_file.is_open());

  std::stringstream tdg_file_stream;
  tdg_file_stream << tdg_file.rdbuf();
  int equal = tdg_string.compare(tdg_file_stream.str());

  assert(equal == 0);

  std::cout << "Passed" << std::endl;
  return 0;
}
// CHECK: Passed
