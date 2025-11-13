// RUN: %libomptarget-compile-generic -fopenmp-version=60
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=DEBUG -check-prefix=CHECK
// REQUIRES: libomptarget-debug

#include <stdio.h>

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
#pragma omp target map(present, alloc : x)
    {
      fprintf(stderr, "In tgt: %d\n", x[1]); // CHECK-NOT: In tgt: 111
      x[1] = 222;
    }
    // DEBUG: omptarget --> Pointer HstPtr=0x[[#%x,HOST_ADDR:]] falls within a
    // DEBUG-SAME:          range previously marked for deletion
    // DEBUG: omptarget --> Moving {{.*}} bytes
    // DEBUG-SAME:          (tgt:0x{{.*}}) -> (hst:0x{{0*}}[[#HOST_ADDR]])
#pragma omp target exit data map(from : p2x[0]) map(delete : p1x[ : ])
    fprintf(stderr, "%d\n", x[1]); // CHECK: 222
  }
}
