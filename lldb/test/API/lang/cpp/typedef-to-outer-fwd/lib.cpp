#include "lib.h"

template <typename T> struct FooImpl {
  using Ref = FooImpl<T> *;

  Ref Create() { return new FooImpl<T>(); }
};

Foo getString() {
  FooImpl<char> impl;
  Foo ret;
  ret.impl = impl.Create();

  return ret;
}
