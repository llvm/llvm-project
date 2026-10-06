#include <stdio.h>

volatile int keep_running = 1;

int main(int argc, char *argv[]) {
  puts("running"); // break here
  fflush(stdout);
  while (keep_running)
    ;
  return 0;
}
