// RUN: %libomptarget-compile-generic
//
// RUN: env LIBOMPTARGET_DEBUG=1 LIBOMPTARGET_INFO=32 \
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=DEBUG
//
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefix=CHECK
//
// REQUIRES: libomptarget-debug

// Ensure that the packed host buffer used to send firstprivate data to the
// device is released only after the transfer is done, i.e. that the buffer
// submitData() reads from is still alive when the queued device operations
// complete.
//
// The firstprivate payload has to be larger than a pointer for the runtime to
// allocate a separate host buffer for it, and small enough to be packed rather
// than transferred on its own, so use an explicitly-sized 48-byte array.

#include <stdint.h>
#include <stdio.h>

#define N 6

int main(void) {
  uint64_t a[N];
  for (int i = 0; i < N; ++i)
    a[i] = i + 1;

  int ok = 0;

  // clang-format off
  // The packed buffer is what gets submitted as the host source, ...
  // DEBUG: Firstprivate array 0x{{.*}} of size 48 will be packed
  // DEBUG: Submitting packed firstprivate arguments from host buffer at 0x[[#%x,BUF:]]
  // DEBUG: Copying data from host to device, HstPtr=0x{{0*}}[[#BUF]], {{.*}}, Size=48
  // ... and it is still alive once the queued operations have completed, which
  // is when the release runs.
  // DEBUG: Synchronization complete, running post-processing
  // DEBUG: Releasing submitData source buffer at 0x{{0*}}[[#BUF]]
  // clang-format on
#pragma omp target firstprivate(a) map(from : ok)
  {
    ok = 1;
    for (int i = 0; i < N; ++i)
      if (a[i] != i + 1)
        ok = 0;
  }

  // CHECK: values_intact=1
  printf("values_intact=%d\n", ok);
  return 0;
}
