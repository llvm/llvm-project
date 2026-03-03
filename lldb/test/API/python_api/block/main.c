#include <stdio.h>

extern int fn(int a, int b);

int main(int argc, char const *argv[]) {
  int a = 3;
  int b = 17;
  int sum = fn(a, b); // breakpoint 1
  printf("fn(3, 17) returns %d\n", sum);
  return 0;
}
