#include "foo.h"

int add(int a, int b) { return a + b; }

int multiply(int a, int b) {
  int result = 0;
  for (int i = 0; i < b; ++i)
    result = add(result, a); // SOURCE THIS LINE
  return result;
}

int factorial(int n) {
  if (n <= 1)
    return 1;
  return multiply(n, factorial(n - 1));
}
