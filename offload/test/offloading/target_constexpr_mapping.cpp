// RUN: %libomptarget-compileoptxx-run-and-check-generic

#include <omp.h>
#include <stdio.h>

#pragma omp declare target
class A {
public:
  constexpr static double pi = 3.141592653589793116;
  A() { ; }
  ~A() { ; }
};
#pragma omp end declare target

#pragma omp declare target
constexpr static double anotherPi = 3.14;
#pragma omp end declare target

int main() {
  double a[2];
#pragma omp target map(tofrom : a[:2])
  {
    a[0] = A::pi;
    a[1] = anotherPi;
  }

  // CHECK: pi = 3.141592653589793116
  printf("pi = %.18f\n", a[0]);

  // CHECK: anotherPi = 3.14
  printf("anotherPi = %.2f\n", a[1]);

  return 0;
}
