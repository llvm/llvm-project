// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=DEBUG -check-prefix=CHECK
// REQUIRES: libomptarget-debug

// The from on target_exit_data should result in a data-transfer of 4 bytes,
// even if when "from" is honored, the ref-count hasn't gone down to 0.
// It will eventually go down to 0 as part of the same exit_data due to the
// "delete" on it.
// This is a case that cannot be handled at compile time because the list-items
// are not related.

#include <stdio.h>

int main() {
  int x[10];
  int *p1x, *p2x;
  p1x = p2x = &x[0];

#pragma omp target data map(alloc : x)
  {
#pragma omp target enter data map(alloc : x) map(to : x)
#pragma omp target map(present, alloc : x)
    {
      printf("In tgt: %d\n", x[1]); // CHECK-NOT: In tgt: 111
      x[1] = 222;
    }
// DEBUG: omptarget --> Found skipped FROM entry
// DEBUG-SAME:          HstPtr=0x[[#%x,HOST_ADDR:]] size=[[#%u,SIZE:]]
// DEBUG-SAME:          within region being deleted
// DEBUG: omptarget --> Moving [[#SIZE]] bytes
// DEBUG-SAME:          (tgt:0x{{.*}}) -> (hst:0x{{0*}}[[#HOST_ADDR]])
#pragma omp target exit data map(delete : p1x[ : ]) map(from : p2x[1])
    printf("%d\n", x[1]); // CHECK: 222
  }
}
