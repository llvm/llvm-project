// RUN: %libomptarget-compile-generic
// RUN: %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: %if libc %{ %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=LIBC %}

// REQUIRES: amdgpu

// Lanes of one wavefront execute a single instruction stream, so if two of
// them contend for the same lock the one that takes it can never run on to
// release it. The runtime traps rather than hanging, after saying why on stderr
// if the GPU libc is available. The lock calls are kept in
// separate functions so that the compiler cannot see the region they enclose
// and serialize the lanes around it.

#include <omp.h>
#include <stdio.h>

#pragma omp declare target
omp_lock_t Lock;
int X;
[[gnu::noinline]] void acquire(void) { omp_set_lock(&Lock); }
[[gnu::noinline]] void release(void) { omp_unset_lock(&Lock); }
#pragma omp end declare target

int main() {
  // LIBC: omp_set_lock: threads of one wavefront are contending for the same lock, which can never be released; aborting
  // CHECK: execution interrupted by hardware trap instruction
#pragma omp target parallel num_threads(64)
  {
#pragma omp single
    omp_init_lock(&Lock);

    acquire();
    X++;
    release();
  }

  // CHECK-NOT: not reached
  printf("not reached\n");
  return 0;
}
