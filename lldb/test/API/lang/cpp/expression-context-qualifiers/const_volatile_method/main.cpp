#include <cassert>
#include <cstdio>

struct Foo {
  double bar() const { return 5.0; }
  const char *bar() volatile { return "volatile_bar"; }
  int bar() volatile const { return 2; }

  int volatile_method() volatile {
    std::puts("Break here: volatile");
    return 0;
  }
  int const_method() const {
    std::puts("Break here: const");
    return 0;
  }
  int other_cv_method() const volatile { return 20; }

  int const_volatile_method() const volatile {
    auto x = bar();
    assert(x == 2);
    other_cv_method();

    std::puts("Break here: const volatile");

    return 120;
  }
};

int main() {
  const Foo f;
  f.bar();
  f.const_method();

  volatile Foo f2;
  f2.bar();
  f2.volatile_method();

  const volatile Foo f3;
  f3.bar();

  return Foo{}.const_volatile_method();
}
