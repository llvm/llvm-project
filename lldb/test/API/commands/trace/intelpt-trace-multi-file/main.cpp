#include "foo.h"

int __attribute__((always_inline)) inline_function() {
  int z = 0;
  z++;
  return z;
}

int main() {
  int res = foo();

  res++;

  res += inline_function();

  res += foo();

  return res;
}
