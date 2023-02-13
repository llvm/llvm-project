#include "lib.h"

struct Foo {};

struct Bar {
  Wrapper<Foo> getWrapper() { return Wrapper<Foo>(); }
  int sinkWrapper(Wrapper<Foo>) { return -1; }
};

int main() {
  Bar b;
  return b.sinkWrapper(b.getWrapper());
}

