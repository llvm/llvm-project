#include "a.h"
#include <cstdio>

bool foo() {
  A a1;
  B b1;

  return b1.member_func_a(a1); // break here
}

int main() {
  int x = 0;

  return foo();
}
