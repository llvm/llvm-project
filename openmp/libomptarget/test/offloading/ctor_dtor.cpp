// RUN: %libomptarget-compilexx-run-and-check-generic
// RUN: %libomptarget-compileoptxx-run-and-check-generic
// RUN: %libomptarget-compilexx-generic && \
// RUN: env OMPTARGET_DUMP_OFFLOAD_ENTRIES=0 %libomptarget-run-generic 2>&1 | \
// RUN: %fcheck-generic --check-prefix=DUMP
//
// DUMP:     Device 0 offload entries:
// DUMP-DAG:   global var.: s
// DUMP-DAG:        kernel: __omp_offloading_{{.*}}_main_
//
#include <cstdio>
struct S {
  S() : i(7) {}
  ~S() { foo(); }
  int foo() { return i; }

private:
  int i;
};

S s;
#pragma omp declare target(s)

int main() {
  int r;
#pragma omp target map(from : r)
  r = s.foo();

  // CHECK: 7
  printf("%i\n", r);
}
