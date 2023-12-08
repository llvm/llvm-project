#include <stdio.h>

void foo() {
  int foo = 10; 
  printf("%d\n", foo); // Set a breakpoint here. 
  foo = 20;
  printf("%d\n", foo);
}

int main() {
  foo();
  return 0;
}
