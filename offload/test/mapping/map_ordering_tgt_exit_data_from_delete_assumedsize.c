// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=CHECK
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=DEBUG
// REQUIRES: libomptarget-debug

// The from on target_exit_data should result in a data-transfer of 4 bytes,
// even if when "delete" is honored first, and by the time "from" is
// encountered, the ref-count had already been 0 (i.e. it's not transitioning
// from non-zero to zero).
// This is a case that cannot be handled at compile time because the list-items
// are not related.

#include <stdio.h>
int main() {
  int x[10];
  int *p1x, *p2x;
  p1x = p2x = &x[1];
  x[1] = 111;

#pragma omp target data map(alloc : x)
  {
#pragma omp target enter data map(alloc : x) map(to : x)
// DEBUG-NOT: omptarget --> Moving {{.*}} bytes (hst:0x{{.*}}) -> (tgt:0x{{.*}})
#pragma omp target map(present, alloc : x)
    {
      // NOTE: It's ok for this to be 111 under "unified_shared_memory"
      printf("In tgt: %d\n", x[1]); // CHECK-NOT: In tgt: 111
      x[1] = 222;
    }

#pragma omp target exit data map(from : p2x[0]) map(delete : p1x[ : ])
    // clang-format off
    // DEBUG: omptarget --> Pointer HstPtr=0x[[#%x,HOST_ADDR:]] falls within a range previously released
    // DEBUG: omptarget --> Moving {{.*}} bytes (tgt:0x{{.*}}) -> (hst:0x{{0*}}[[#HOST_ADDR]])
    // clang-format on

    printf("%d\n", x[1]); // CHECK: 222
  }
}
