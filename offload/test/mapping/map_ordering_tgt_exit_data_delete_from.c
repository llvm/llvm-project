// RUN: %libomptarget-compile-run-and-check-generic

#include <stdio.h>

int main() {
  int x = 111;
#pragma omp target data map(alloc : x)
  {
#pragma omp target enter data map(alloc : x) map(to : x)
#pragma omp target map(present, alloc : x)
    {
      // NOTE: It's ok for this to be 111 under "unified_shared_memory"
      printf("%d\n", x); // CHECK-NOT: 111
      x = 222;
    }
#pragma omp target exit data map(delete : x) map(from : x) map(delete : x)
    printf("%d\n", x); // CHECK: 222
  }
}
