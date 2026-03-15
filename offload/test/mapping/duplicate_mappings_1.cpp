// clang-format off
// RUN: %libomptarget-compilexx-generic -Wno-openmp-mapping && %libomptarget-run-generic
// XFAIL: intelgpu

// clang-format on

#include <assert.h>

struct Inner {
  int *data;
  Inner(int size) { data = new int[size](); }
  ~Inner() { delete[] data; }
};

struct Outer {
  Inner i;
  Outer() : i(10) {}
};

int main() {
  Outer o;
#pragma omp target map(tofrom : o.i.data[0 : 10]) map(tofrom : o.i.data[0 : 10])
  {
    o.i.data[0] = 42;
  }
  assert(o.i.data[0] == 42);
  return 0;
}
