// Simple test program with a few stack frames

#include <stdio.h>

int foo(int x) {
  printf("In foo: %d\n", x); // Break here
  return x * 2;
}

int main(int argc, char **argv) {
  int result = foo(42);
  printf("Result: %d\n", result);
  return 0;
}