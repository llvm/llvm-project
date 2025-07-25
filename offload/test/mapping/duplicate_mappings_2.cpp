// clang-format off
// RUN: %libomptarget-compilexx-generic -Wno-openmp-mapping && %libomptarget-run-generic

#include <assert.h>

// clang-format on

struct Inner {
  int *data;
  Inner(int size) { data = new int[size](); }
  ~Inner() { delete[] data; }
};
#pragma omp declare mapper(Inner i) map(i, i.data[0 : 10])

struct Outer {
  Inner i;
  Outer() : i(10) {}
};
#pragma omp declare mapper(Outer o) map(o, o.i)

int main() {
  Outer o;
#pragma omp target map(tofrom : o)
  {
    o.i.data[0] = 42;
  }
  assert(o.i.data[0] == 42);
  return 0;
}
