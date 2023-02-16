#include "lib.h"

struct Bar {
  Wrapper<Foo> getWrapper() { return Wrapper<Foo>(); }
  int sinkWrapper(Wrapper<Foo>) { return -1; }
};

int main() {
  Bar b;
  Wrapper<int> w1;
  Wrapper<double> w2;
  Wrapper<Foo> w3 = getFooWrapper();
  Wrapper<Foo> w4;
  return b.sinkWrapper(b.getWrapper());
}

