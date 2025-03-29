// RUN: %libomptarget-compilexx-generic
// RUN: %libomptarget-run-generic
// RUN: %libomptarget-compilexx-generic -DCTOR_KERNEL
// RUN: %not --crash %libomptarget-run-generic
// RUN: %libomptarget-compilexx-generic -DCTOR_API
// RUN: %not --crash %libomptarget-run-generic

#include <cstdio>
#include <omp.h>

void foo_dev() { __builtin_trap(); }

#pragma omp declare variant(foo_dev) match(device = {kind(nohost)})
void foo() {}

struct S {
  S() { foo(); }
};

S s;
#pragma omp declare target(s)

int main() {
  int Dev = omp_get_default_device();

#ifdef CTOR_KERNEL
#pragma omp target
  {}
#endif
#ifdef CTOR_API
  omp_get_mapped_ptr(&s, Dev);
#endif
}
