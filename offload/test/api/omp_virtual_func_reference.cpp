// RUN: %libomptarget-compilexx-run-and-check-generic

#include <assert.h>
#include <omp.h>
#include <stdio.h>

#define TEST_VAL 10

#pragma omp declare target
class Base {
public:
  virtual int foo(int x) { return x; }
};

class Derived : public Base {
public:
  virtual int foo(int x) { return -x; }
};
#pragma omp end declare target

int test_virtual_reference() {
  Derived ddd;
  Base cont;
  Base &bbb = ddd;

  int b_ret, d_ret, c_ret;

#pragma omp target data map(to : ddd, cont)
  {
#pragma omp target map(bbb, ddd, cont) map(from : b_ret, d_ret, c_ret)
    {
      b_ret = bbb.foo(TEST_VAL);
      d_ret = ddd.foo(TEST_VAL);
      c_ret = cont.foo(TEST_VAL);
    }
  }

  assert(c_ret == TEST_VAL && "Control Base call failed on gpu");
  assert(b_ret == -TEST_VAL && "Control Base call failed on gpu");
  assert(d_ret == -TEST_VAL && "Derived call failed on gpu");

  return 0;
}

int test_virtual_reference_implicit() {
  Derived ddd;
  Base cont;
  Base &bbb = ddd;

  int b_ret, d_ret, c_ret;

#pragma omp target data map(to : ddd, cont)
  {
#pragma omp target map(from : b_ret, d_ret, c_ret)
    {
      b_ret = bbb.foo(TEST_VAL);
      d_ret = ddd.foo(TEST_VAL);
      c_ret = cont.foo(TEST_VAL);
    }
  }

  assert(c_ret == TEST_VAL && "Control Base call failed on gpu");
  assert(b_ret == -TEST_VAL && "Control Base call failed on gpu");
  assert(d_ret == -TEST_VAL && "Derived call failed on gpu");

  return 0;
}

int main() {
  test_virtual_reference();
  test_virtual_reference_implicit();

  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
