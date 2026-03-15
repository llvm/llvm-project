#include "lib.h"

template <typename T> struct BarImpl {
  using Ref = BarImpl<T> *;

  Ref Create() { return new BarImpl<T>(); }
};

BarImpl<char> gMainLocalDef;
FooImpl<char> *gMainExternalDef = nullptr;

int main() { return 0; }
