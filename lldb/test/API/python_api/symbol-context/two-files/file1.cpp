#include "decls.h"

int g() {
  return 1;
}

struct1::~struct1() {
  int x = g(); // Break1
}

void struct1::f() {}

int main() {
  struct1::f();
  struct2::f();

  struct1 s1;
  struct2 s2;

  return 0;
}
