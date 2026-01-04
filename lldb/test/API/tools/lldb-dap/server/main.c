#include <stdio.h>

int main(int argc, char const *argv[]) {
  if (argc == 2) { // breakpoint 1
    printf("Hello %s!\n", argv[1]);
  } else {
    printf("Hello World!\n");
  }
  return 0;
}
