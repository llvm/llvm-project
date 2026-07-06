#include <stdio.h>

int do_something(int input) {
  return input % 5; // Deletable location
}

int main() {
  printf("Set a breakpoint here.\n");
  do_something(100);
  do_something(200);
  return 0;
}
