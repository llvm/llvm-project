#include "subdir/foo.h"
#include <cstdio>

int main() {
  int sum = add(3, 4);
  int prod = multiply(5, 6);
  int fact = factorial(5);

  printf("add(3, 4) = %d\n", sum);
  printf("multiply(5, 6) = %d\n", prod); // SOURCE THIS LINE
  printf("factorial(5) = %d\n", fact);

  return 0;
}
