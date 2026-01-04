#include <stdio.h>

int c(int val) {
  // Set break point at this line
  return val + 3;
}

int b(int val) {
  int result = c(val);
  return result;
}

int a(int val) {
  int result = b(val);
  return result;
}

int main() {
  int result = a(1);
  printf("Result: %d\n", result);
  return 0;
}
