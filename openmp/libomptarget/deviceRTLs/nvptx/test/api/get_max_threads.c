// RUN: %compile-run-and-check
#include <omp.h>
#include <stdio.h>

int main(){
  int max_threads = -1;
  int num_threads = -1;

  #pragma omp target map(tofrom: max_threads)
    max_threads = omp_get_max_threads();

  #pragma omp target parallel map(tofrom: num_threads)
  {
    #pragma omp master
      num_threads = omp_get_num_threads();
  }
  
  // CHECK: Max Threads: 128, Num Threads: 128
  printf("Max Threads: %d, Num Threads: %d\n", max_threads, num_threads);

  return 0;
}
