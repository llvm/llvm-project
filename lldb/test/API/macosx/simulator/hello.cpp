#include <stdio.h>
#include <thread>
#include <unistd.h>

static void print_pid() { fprintf(stderr, "PID: %d\n", getpid()); }

// The test kills this process after the `platform process list` check, so this
// sleep should never expire.
static void sleep() { std::this_thread::sleep_for(std::chrono::seconds(600)); }

int main(int argc, char **argv) {
  print_pid();
  puts("break here\n");
  sleep();
  return 0;
}
