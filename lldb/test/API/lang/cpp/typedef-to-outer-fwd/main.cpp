#include "lib.h"

extern Foo getString();

int main() {
  FooImpl<char> *foo = getString().impl;
  return 0;
}
