#include "header.h"

int main() {
  int a = 3;
  int b = 4;
  int c = c = add(a, b);
  int d = mul(a, b);
  return equal(a, sub(c, b)) - equal(a, div(d, b));
}
