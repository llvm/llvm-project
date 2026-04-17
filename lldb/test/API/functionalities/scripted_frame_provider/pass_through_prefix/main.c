#include <stdio.h>

int baz() {
  printf("baz\n");
  return 42; // break here.
}

int bar() {
  printf("bar\n");
  return baz();
}

int foo() {
  printf("foo\n");
  return bar();
}

int main() {
  printf("main\n");
  return foo();
}
