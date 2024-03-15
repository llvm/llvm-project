// clang-format off
// RUN: %clang-generic -fPIC -shared %S/../Inputs/declare_indirect_func.c -o %T/libslfm.so  -fopenmp-version=51
// RUN: %libomptarget-compile-generic -rpath %T -L %T -l slfm -o %t  -fopenmp-version=51
// RUN: env LIBOMPTARGET_INFO=32 %t 2>&1 | %fcheck-generic
// clang-format on

#include <stdio.h>

extern int func(); // Provided in liba.so, returns 42
typedef int (*fp_t)();

int main() {
  int x = 0;
  fp_t fp = &func;
  printf("TARGET\n");
#pragma omp target map(from : x)
  x = fp();
  // CHECK: Copying data from device to host, {{.*}} Size=8
  // CHECK: Copying data from device to host, {{.*}} Size=4
  // CHECK: 42
  printf("%i\n", x);
}
