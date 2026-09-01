#include <stdio.h>

int foo(int x) {
  int y = x + 1; // break here
  return y;
}

int main() {
  int result = foo(41);
  printf("%d\n", result);
  return 0;
}
