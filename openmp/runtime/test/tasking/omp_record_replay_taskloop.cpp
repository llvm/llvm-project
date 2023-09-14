// RUN: %libomp-cxx-compile-and-run
#include <iostream>
#include <cassert>

#define NT 20
#define N 128*128

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

int main() {
  int num_tasks = 0;

  int array[N];
  for (int i = 0; i < N; ++i)
    array[i] = 1;

  long sum = 0;
  #pragma omp parallel
  #pragma omp single
  for (int iter = 0; iter < NT; ++iter) {
    #pragma ompx taskgraph
    {
      num_tasks++;
      #pragma omp taskloop reduction(+:sum) num_tasks(4096)
      for (int i = 0; i < N; ++i) {
        sum += array[i];
      }
    }
  }
  assert(sum==N*NT);
  assert(num_tasks==1);

  std::cout << "Passed" << std::endl;
  return 0;
}
// CHECK: Passed
