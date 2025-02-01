#include <stdio.h>
#include <stdlib.h>

#include <threads.h>
#include <time.h>

int main(int argc, char *argv[]) {
  const char *foo = getenv("FOO");
  for (int counter = 1;; counter++) {
    thrd_sleep(&(struct timespec){.tv_sec = 1}, NULL); // breakpoint
  }
  return 0;
}
