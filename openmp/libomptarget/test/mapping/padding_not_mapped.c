// RUN: %libomptarget-compile-generic -fopenmp-version=51
// RUN: %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic

// The host memory layout for the following program looks like this:
//
//   | 4 bytes | 4 bytes | 8 bytes |
//   |   s.x   |   s.y   |   s.z   |
//   `-----------------------------'
//
// s is always at least 8-byte aligned in host memory due to s.z, so
// libomptarget's device padding for map(s.y,s.z) always maps to host memory
// that includes s.x.  At one time, s.x appeared to be mapped as a result, but
// libomptarget has since been fixed not to consider device padding as mapped to
// host memory.

#include <omp.h>
#include <stdio.h>

int main() {
  struct S { int x; int y; double z; } s = {1, 2, 3};

  // CHECK: &s.x = 0x[[#%x,HOST_ADDR:]], size = [[#%u,SIZE:]]
  fprintf(stderr, "&s = %p\n", &s);
  fprintf(stderr, "&s.x = %p, size = %ld\n", &s.x, sizeof s.x);
  fprintf(stderr, "&s.y = %p\n", &s.y);
  fprintf(stderr, "&s.z = %p\n", &s.z);

  // CHECK: s.x is present: 0
  // CHECK: s.x = 1{{$}}
  #pragma omp target enter data map(alloc: s.y, s.z)
  int dev = omp_get_default_device();
  fprintf(stderr, "s.x is present: %d\n", omp_target_is_present(&s.x, dev));
  #pragma omp target update from(s.x) // should have no effect
  fprintf(stderr, "s.x = %d\n", s.x);

  // CHECK: Libomptarget message: device mapping required by 'present' map type modifier does not exist for host address 0x{{0*}}[[#HOST_ADDR]] ([[#SIZE]] bytes)
  // CHECK: Libomptarget error: Call to getTargetPointer returned null pointer ('present' map type modifier).
  // CHECK: Libomptarget fatal error 1: failure of target construct while offloading is mandatory
  #pragma omp target enter data map(present, alloc: s.x)

  return 0;
}
