// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=DEBUG -check-prefix=CHECK
// REQUIRES: libomptarget-debug

// There should only be one "from" data-transfer, despite the two duplicate
// maps.

#include <stdio.h>

int main() {
  int x = 111;
#pragma omp target data map(alloc : x)
  {
#pragma omp target enter data map(alloc : x) map(to : x)
#pragma omp target map(present, alloc : x)
    {
      printf("In tgt: %d\n", x); // CHECK-NOT: In tgt: 111
      x = 222;
    }
#pragma omp target exit data map(always, from : x) map(always, from : x)
    // DEBUG: omptarget --> Moving {{.*}} bytes (tgt:0x{{.*}}) -> (hst:0x{{.*}})
    // DEBUG-NOT: omptarget --> Moving {{.*}} bytes
  }

  printf("%d\n", x); // CHECK: 222
}
