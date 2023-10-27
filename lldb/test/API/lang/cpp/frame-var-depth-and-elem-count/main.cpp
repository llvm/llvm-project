#include <cstdio>

struct A {
  int i = 42;
};

struct B {
  A a;
};

struct C {
  B b;
};

int main() {
  C *c = new C[5];
  puts("break here");
  return 0;
}
