// RUN: %libomptarget-compile-generic -DFIRST -c -o %t.first.o
// RUN: %libomptarget-compile-generic -DSECOND -c -o %t.second.o
// RUN: not %clang-generic %t.second.o %t.first.o -o %t 2>&1 | \
// %fcheck-plain-generic %s
//
// REQUIRES: gpu
//
// CHECK: multiple definition

#include <stdio.h>

#ifdef FIRST
void first(void) {
  int x = 0;
#pragma omp target ompx_name("duplicate_link_kernel") map(tofrom : x)
  {
    x = 1;
  }
  printf("x: %i\n", x);
}
#endif

#ifdef SECOND
void second(void) {
  int x = 0;
#pragma omp target ompx_name("duplicate_link_kernel") map(tofrom : x)
  {
    x = 2;
  }
  printf("x: %i\n", x);
}

void first(void);

int main(void) {
  first();
  second();
  return 0;
}
#endif
