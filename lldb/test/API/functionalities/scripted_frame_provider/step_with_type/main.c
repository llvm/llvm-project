#include <stdio.h>

void bar() {
  printf("bar\n");
  return;
}

int g_counter = 0;

void foo() {
  bar();
  g_counter++;
  printf("foo\n");
  return;
}

int main() {
  printf("main\n");

  foo(); // Stop here to step
  foo();
  foo();
  foo();
  foo();
  foo();

  return 0;
}
