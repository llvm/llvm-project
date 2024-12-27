#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
  // Ensure multiple partial lines are detected and sent.
  printf("abc");
  printf("def");
  printf("ghi\n");
  printf("hello world\n"); // breakpoint 1
  return 0;
}
