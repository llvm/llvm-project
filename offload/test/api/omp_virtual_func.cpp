// RUN: %libomptarget-compilexx-run-and-check-generic
#include <assert.h>
#include <omp.h>
#include <stdio.h>

#define TEST_VAL 10

#pragma omp declare target

class Base {
public:
  virtual int foo() { return 1; }
  virtual int bar() { return 2; }
  virtual int foo_with_arg(int x) { return x; }
};

class Derived : public Base {
public:
  virtual int foo() { return 10; }
  virtual int bar() { return 20; }
  virtual int foo_with_arg(int x) { return -x; }
};

#pragma omp end declare target

int test_virtual_implicit_map() {
  Base base;
  Derived derived;
  int result1, result2, result3, result4, result5, result6;

  // map both base and derived objects up front, since the spec
  // requires that when first mapping a C++ object that the static
  // type must match the dynamic type
#pragma omp target data map(base, derived)
  {
    Base *p1 = &base;
    Base *p2 = &derived;

#pragma omp target map(from : result1, result2, result3, result4, result5,     \
                           result6)
    {
      // These calls will fail if Clang does not
      // translate/attach the vtable pointer in each object
      result1 = p1->foo();
      result2 = p1->bar();
      result3 = p2->foo();
      result4 = p2->bar();
      result5 = base.foo();
      result6 = derived.foo();
    }
  }

  assert(result1 == 1 && "p1->foo() implicit map Failed");
  assert(result2 == 2 && "p1->bar() implicit map Failed");
  assert(result3 == 10 && "p2->foo() implicit map Failed");
  assert(result4 == 20 && "p2->bar() implicit map Failed");
  assert(result5 == 1 && "base.foo() implicit map Failed");
  assert(result6 == 10 && "derived.foo() implicit map Failed");
  return 0;
}

int test_virtual_explicit_map() {
  Base base;
  Derived derived;
  int result1, result2, result3, result4;

  // map both base and derived objects up front, since the spec
  // requires that when first mapping a C++ object that the static
  // type must match the dynamic type
#pragma omp target data map(base, derived)
  {
    Base *p1 = &base;
    Base *p2 = &derived;

#pragma omp target map(p1[0 : 0], p2[0 : 0])                                   \
    map(from : result1, result2, result3, result4)
    {
      result1 = p1->foo();
      result2 = p1->bar();
      result3 = p2->foo();
      result4 = p2->bar();
    }
  }

  assert(result1 == 1 && "p1->foo() explicit map Failed");
  assert(result2 == 2 && "p1->bar() explicit map Failed");
  assert(result3 == 10 && "p2->foo() explicit map Failed");
  assert(result4 == 20 && "p2->bar() explicit map Failed");
  return 0;
}

int test_virtual_reference() {
  Derived ddd;
  Base cont;
  Base &bbb = ddd;

  int b_ret, d_ret, c_ret;

#pragma omp target data map(to : ddd, cont)
  {
#pragma omp target map(bbb, ddd, cont) map(from : b_ret, d_ret, c_ret)
    {
      b_ret = bbb.foo_with_arg(TEST_VAL);
      d_ret = ddd.foo_with_arg(TEST_VAL);
      c_ret = cont.foo_with_arg(TEST_VAL);
    }
  }

  assert(c_ret == TEST_VAL && "Control Base call failed on gpu");
  assert(b_ret == -TEST_VAL && "Reference to derived call failed on gpu");
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
      b_ret = bbb.foo_with_arg(TEST_VAL);
      d_ret = ddd.foo_with_arg(TEST_VAL);
      c_ret = cont.foo_with_arg(TEST_VAL);
    }
  }

  assert(c_ret == TEST_VAL && "Control Base call failed on gpu (implicit)");
  assert(b_ret == -TEST_VAL &&
         "Reference to derived call failed on gpu (implicit)");
  assert(d_ret == -TEST_VAL && "Derived call failed on gpu (implicit)");

  return 0;
}

int main() {
  test_virtual_implicit_map();
  test_virtual_explicit_map();
  test_virtual_reference();
  test_virtual_reference_implicit();

  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
