#include <limits.h>

int main() {
  volatile int a = INT_MAX;
  volatile int b = 1;
  volatile int c = a + b;
  return c;
}