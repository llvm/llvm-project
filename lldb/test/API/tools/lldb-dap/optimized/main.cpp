#include <stdio.h>
#include <string>

int foo(int x, int y) {
  printf("Got input %d, %d\n", x, y);
  return x + y + 3; // breakpoint 1
}

int main(int argc, char const *argv[]) {
  printf("argc: %d\n", argc);
  int result = foo(20, argv[0][0]);
  printf("result: %d\n", result); // breakpoint 2
  return 0;
}
