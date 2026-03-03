#include <stdio.h>

int fn(int a, int b) {
  if (a < b) {
    int sum = a + b;
    return sum; // breakpoint 2
  }

  return a * b;
}

int main(int argc, char const *argv[]) {
  int a = 3;
  int b = 17;
  int sum = fn(a, b); // breakpoint 1
  printf("fn(3, 17) returns %d\n", sum);
  return 0;
}
