#undef DEF
#include "header.h"

static int foo() {
  return 0;
}

int main() {
  f1();

  long *x;
  f2(&x);

  double *y;
  f2(&y);

  f3();

  return foo();
}
