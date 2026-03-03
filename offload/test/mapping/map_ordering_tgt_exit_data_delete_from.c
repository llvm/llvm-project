// RUN: %libomptarget-compile-run-and-check-generic
// XFAIL: intelgpu

#include <stdio.h>

int main() {
  int x = 111;
#pragma omp target data map(alloc : x)
  {
#pragma omp target enter data map(alloc : x) map(to : x)
#pragma omp target map(present, alloc : x)
    {
      // NOTE: It's ok for this to be 111 under "unified_shared_memory"
      printf("In tgt: %d\n", x); // CHECK-NOT: In tgt: 111
      x = 222;
    }
#pragma omp target exit data map(delete : x) map(from : x) map(delete : x)
    printf("After tgt exit data: %d\n", x); // CHECK: After tgt exit data: 222
  }
}
