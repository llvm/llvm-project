// RUN: %libomptarget-compile-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-x86_64-pc-linux-gnu

// Clang 6.0 doesn't use the new map interface, undefined behavior when
// the compiler emits "old" interface code for structures.
// UNSUPPORTED: clang-6

#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int *ptr1;
  int *ptr2;
} StructWithPtrs;

int main(int argc, char *argv[]) {
  StructWithPtrs s, s2;
  s.ptr1 = malloc(sizeof(int));
  s.ptr2 = malloc(2 * sizeof(int));
  s2.ptr1 = malloc(sizeof(int));
  s2.ptr2 = malloc(2 * sizeof(int));

#pragma omp target enter data map(to: s2.ptr2[0:1])
#pragma omp target map(s.ptr1[0:1], s.ptr2[0:2])
  {
    s.ptr1[0] = 1;
    s.ptr2[0] = 2;
    s.ptr2[1] = 3;
  }
#pragma omp target exit data map(from: s2.ptr1[0:1], s2.ptr2[0:1])

  // CHECK: s.ptr1[0] = 1
  // CHECK: s.ptr2[0] = 2
  // CHECK: s.ptr2[1] = 3
  printf("s.ptr1[0] = %d\n", s.ptr1[0]);
  printf("s.ptr2[0] = %d\n", s.ptr2[0]);
  printf("s.ptr2[1] = %d\n", s.ptr2[1]);

  free(s.ptr1);
  free(s.ptr2);
  free(s2.ptr1);
  free(s2.ptr2);

  return 0;
}
