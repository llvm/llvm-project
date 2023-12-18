#include <stdio.h>

int main(int argc, char const *argv[], char const *envp[]) {
  int i = 0;
  printf("Do something\n"); // breakpoint A
  printf("Do something else\n");
  i = 1234;
  return 0; // breakpoint B
}
