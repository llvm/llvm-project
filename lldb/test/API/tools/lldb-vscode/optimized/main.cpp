#include <stdio.h>
#include <string>

int foo(int x, int y) {
  printf("Got input %d, %d\n", x, y);
  return x + y + 3; // breakpoint 1
}

int main(int argc, char const *argv[]) {
  int optimized = argc > 1 ? std::stoi(argv[1]) : 0;

  printf("argc: %d, optimized: %d\n", argc, optimized);
  int result = foo(argc, 20);
  printf("result: %d\n", result); // breakpoint 2
  return 0;
}
