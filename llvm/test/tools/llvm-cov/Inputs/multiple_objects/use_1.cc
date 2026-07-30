#define DEF
#include "header.h"

int main() {
  f1();

  int *x;
  f2(&x);

  float *y;
  f2(&y);

  return 0;
}
