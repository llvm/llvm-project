// RUN: %libomptarget-compilexx-and-run-generic

#include <cassert>
#include <iostream>
#include <stdexcept>

int main(int argc, char *argv[]) {
  int a = 0;
  std::cout << "outside a = " << a << " addr " << &a << std::endl;
#pragma omp target map(tofrom : a) depend(out : a) nowait
  {
    int sum = 0;
    for (int i = 0; i < 100000; i++)
      sum++;
    a = 1;
  }

#pragma omp task depend(inout : a) shared(a)
  {
    std::cout << "a = " << a << " addr " << &a << std::endl;
    if (a != 1)
      throw std::runtime_error("wrong result!");
    a = 2;
  }

#pragma omp task depend(inout : a) shared(a)
  {
    std::cout << "a = " << a << " addr " << &a << std::endl;
    if (a != 2)
      throw std::runtime_error("wrong result!");
    a = 3;
  }

#pragma omp taskwait

  assert(a == 3 && "wrong result!");

  return 0;
}
