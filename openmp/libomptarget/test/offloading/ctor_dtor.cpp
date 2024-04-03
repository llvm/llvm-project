// RUN: %libomptarget-compilexx-run-and-check-generic
// RUN: %libomptarget-compileoptxx-run-and-check-generic

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
