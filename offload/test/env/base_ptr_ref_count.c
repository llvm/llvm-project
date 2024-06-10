// RUN: %libomptarget-compile-generic && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 | %fcheck-generic
// REQUIRES: libomptarget-debug

#include <stdio.h>
#include <stdlib.h>

int *allocate(size_t n) {
  int *ptr = malloc(sizeof(int) * n);
#pragma omp target enter data map(to : ptr[ : n])
  return ptr;
}

void deallocate(int *ptr, size_t n) {
#pragma omp target exit data map(delete : ptr[ : n])
  free(ptr);
}

#pragma omp declare target
int *cnt;
void foo() { ++(*cnt); }
#pragma omp end declare target

int main(void) {
  int *A = allocate(10);
  int *V = allocate(10);
  deallocate(A, 10);
  deallocate(V, 10);
  // CHECK-NOT: RefCount=2
  cnt = malloc(sizeof(int));
  *cnt = 0;
#pragma omp target map(cnt[ : 1])
  foo();
  printf("Cnt = %d.\n", *cnt);
  // CHECK: Cnt = 1.
  *cnt = 0;
#pragma omp target data map(cnt[ : 1])
#pragma omp target
  foo();
  printf("Cnt = %d.\n", *cnt);
  // CHECK: Cnt = 1.
  free(cnt);

  return 0;
}
