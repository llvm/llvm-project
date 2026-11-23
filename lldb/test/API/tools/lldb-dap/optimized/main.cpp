#include <stdio.h>
#include <string>

int foo(int x, int y) {
  printf("Got input %d, %d\n", x, y);
  return x + y + 3; // breakpoint 1
}

__attribute__((noinline)) void use(int v) { printf("%d\n", v); }

__attribute__((noinline)) void constant_local() {
  int k = 42;
  use(k);
  use(k + 1); // breakpoint 3
}

int main(int argc, char const *argv[]) {
  printf("argc: %d\n", argc);
  int result = foo(20, argv[0][0]);
  printf("result: %d\n", result); // breakpoint 2
  constant_local();
  return 0;
}
