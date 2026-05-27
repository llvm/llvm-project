// RUN: %libomptarget-compile-run-and-check-generic

#include <assert.h>
#include <stdio.h>

#pragma omp begin declare variant match(device = {kind(gpu)})
// Provided by the runtime.
void *__llvm_omp_indirect_call_lookup(void *host_ptr);
#pragma omp declare target to(__llvm_omp_indirect_call_lookup)                 \
    device_type(nohost)
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(cpu)})
// We assume unified addressing on the CPU target.
void *__llvm_omp_indirect_call_lookup(void *host_ptr) { return host_ptr; }
#pragma omp end declare variant

#pragma omp begin declare target indirect
void foo(int *x) { *x = *x + 1; }
void bar(int *x) { *x = *x + 1; }
void baz(int *x) { *x = *x + 1; }
#pragma omp end declare target

int main() {
  void *foo_ptr = foo;
  void *bar_ptr = bar;
  void *baz_ptr = baz;

  int count = 0;
  void *foo_res;
  void *bar_res;
  void *baz_res;
#pragma omp target map(to : foo_ptr, bar_ptr, baz_ptr) map(tofrom : count)
  {
    foo_res = __llvm_omp_indirect_call_lookup(foo_ptr);
    ((void (*)(int *))foo_res)(&count);
    bar_res = __llvm_omp_indirect_call_lookup(bar_ptr);
    ((void (*)(int *))bar_res)(&count);
    baz_res = __llvm_omp_indirect_call_lookup(baz_ptr);
    ((void (*)(int *))baz_res)(&count);
  }

  assert(count == 3 && "Calling failed");

  // CHECK: PASS
  printf("PASS\n");
}
