// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

#include <assert.h>
#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

int main(int argc, char *argv[]) {
  int dev = omp_get_default_device();
  int x = 10;
  int *x_dev = (int *)omp_target_alloc(sizeof x, dev);
  assert(x_dev && "expected omp_target_alloc to succeed");
  int rc = omp_target_associate_ptr(&x, x_dev, sizeof x, 0, dev);
  assert(!rc && "expected omp_target_associate_ptr to succeed");

// To determine whether x needs to be transferred, the runtime cannot simply
// check whether unified shared memory is enabled and the 'close' modifier is
// specified.  It must check whether x was previously placed in device memory
// by, for example, omp_target_associate_ptr.
#pragma omp target map(always, tofrom : x)
  x += 1;

  // CHECK: x=11
  printf("x=%d\n", x);
  // CHECK: present: 1
  printf("present: %d\n", omp_target_is_present(&x, dev));

  return 0;
}
