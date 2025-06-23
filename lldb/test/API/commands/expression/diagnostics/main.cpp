#include "lib.h"

void foo(int x) {}

struct FooBar {
  int i;
};

enum class EnumInSource { A, B, C };

template <typename T> void templateFunc() {}

int main() {
  FooBar f;
  foo(1);
  foo('c', "abc");
  foo();
  EnumInSource e = EnumInSource::A;
  templateFunc<int>();

  return 0; // Break here
}
