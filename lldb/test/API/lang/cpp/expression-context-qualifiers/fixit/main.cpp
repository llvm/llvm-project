#include <cassert>
#include <cstdio>

struct Bar {
  void method() {}
};

struct Foo {
  int const_method() const {
    std::puts("Break here");

    return 120;
  }

  Bar m_bar;
};

int main() {
  Foo{}.m_bar.method();

  return Foo{}.const_method();
}
