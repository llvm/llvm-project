// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=DEBUG -check-prefix=CHECK
// REQUIRES: libomptarget-debug

#include <stdio.h>

// Even if the "alloc" and "from" are encountered before the "to",
// there should be a data-transfer from host to device, as the
// ref-count goes from 0 to 1 at the entry of the target region.

int main() {
  int x = 111;
  // clang-format off
  // DEBUG: omptarget --> HstPtrBegin 0x[[#%x,HOST_ADDR:]] was newly allocated for the current region
  // DEBUG: omptarget --> Moving {{.*}} bytes (hst:0x{{0*}}[[#HOST_ADDR]]) -> (tgt:0x{{.*}})
  // clang-format on
#pragma omp target map(alloc : x) map(from : x) map(to : x) map(alloc : x)
  {
    printf("%d\n", x); // CHECK: 111
    x = x + 111;
  }

  printf("%d\n", x); // CHECK: 222
}
