#include <cassert>
#include <cstdio>

struct Foo {
  double bar() { return 5.0; }

  int bar() const { return 2; }

  void non_const_method() {}

  int template_const_method() const {
    auto x = bar();
    assert(x == 2);
    std::puts("Break: const_method begin");

    [x] {
      std::puts("Keep on multiple lines...");
      std::puts("Break: const_method no-this lambda");
    }();

    [x]() mutable {
      std::puts("Keep on multiple lines...");
      std::puts("Break: const_method mutable no-this lambda");
    }();

    [this, y = x] {
      auto x = bar() + y;
      std::puts("Break: const_method lambda");
    }();

    [this, y = x]() mutable {
      auto x = bar() + y;
      std::puts("Break: const_method mutable lambda");
    }();

    return 120;
  }

  float m_mem = -2.0;
  const float m_const_mem = -3.0;
};

int main() {
  const Foo f;
  f.bar();

  Foo f2;
  f2.bar();

  return Foo{}.template_const_method();
}
