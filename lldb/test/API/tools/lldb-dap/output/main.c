#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
  // Ensure multiple partial lines are detected and sent.
  printf("abc");
  printf("def");
  printf("ghi\n");
  printf("hello world\n"); // breakpoint 1
  // Ensure the OutputRedirector does not consume the programs \0\0 output.
  char buf[] = "finally\0";
  write(STDOUT_FILENO, buf, sizeof(buf));
  return 0;
}
