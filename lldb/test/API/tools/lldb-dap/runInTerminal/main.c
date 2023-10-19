#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  const char *foo = getenv("FOO");
  for (int counter = 1;; counter++) {
    sleep(1); // breakpoint
  }
  return 0;
}
