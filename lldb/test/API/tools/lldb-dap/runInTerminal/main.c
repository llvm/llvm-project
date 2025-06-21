#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  const char *foo = getenv("FOO");
  int counter = 1;

  return 0; // breakpoint
}
