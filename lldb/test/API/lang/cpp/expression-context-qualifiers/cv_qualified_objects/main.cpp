#include <cstdio>

struct Foo {
  double bar() { return 5.0; }
  int bar() const { return 2; }
  short bar() volatile { return 8; }
  char const *bar() const volatile { return "volatile"; }

  float m_mem = -2.0;
  const float m_const_mem = -3.0;
};

int main() {
  Foo f;
  const Foo cf;
  volatile Foo vf;
  const volatile Foo cvf;

  f.bar();
  cf.bar();
  vf.bar();
  cvf.bar();

  std::puts("Break here");
}
