// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

struct S {
  int x;
  int y;
};

int main(int argc, char *argv[]) {
  int dev = omp_get_default_device();
  struct S s = {10, 20};

  #pragma omp target enter data map(close, to: s)
  #pragma omp target map(alloc: s)
  {
    s.x = 11;
    s.y = 21;
  }
  // To determine whether x needs to be transfered or deleted, the runtime
  // cannot simply check whether unified shared memory is enabled and the
  // 'close' modifier is specified.  It must check whether x was previously
  // placed in device memory by, for example, a 'close' modifier that isn't
  // specified here.  The following struct member case checks a special code
  // path in the runtime implementation where members are transferred before
  // deletion of the struct.
  #pragma omp target exit data map(from: s.x, s.y)

  // CHECK: s.x=11, s.y=21
  printf("s.x=%d, s.y=%d\n", s.x, s.y);
  // CHECK: present: 0
  printf("present: %d\n", omp_target_is_present(&s, dev));

  return 0;
}
