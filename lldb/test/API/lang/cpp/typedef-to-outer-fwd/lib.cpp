#include "lib.h"

template <typename T> struct FooImpl {
  using Ref = FooImpl<T> *;

  Ref Create() { return new FooImpl<T>(); }
};

FooImpl<char> gLibLocalDef;
BarImpl<char> *gLibExternalDef = nullptr;
