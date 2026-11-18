#include <stdio.h>

void foo(void);

int main(void) {
  foo();
  return 0;
}

void foo(void) {
  int (*main_ptr)(void) = main;
  printf("%p\n", (void *)main_ptr); // break here
}
