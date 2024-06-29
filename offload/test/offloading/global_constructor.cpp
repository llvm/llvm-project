// RUN: %libomptarget-compilexx-generic && %libomptarget-run-generic | %fcheck-generic

#include <cstdio>

int foo() { return 1; }

class C {
public:
  C() : x(foo()) {}

  int x;
};

C c;
#pragma omp declare target(c)

int main() {
  int x = 0;
#pragma omp target map(from : x)
  { x = c.x; }

  // CHECK: PASS
  if (x == 1)
    printf("PASS\n");
}
