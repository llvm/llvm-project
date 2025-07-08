#include <stdio.h>
#include <thread>
#include <unistd.h>

static void print_pid() { fprintf(stderr, "PID: %d\n", getpid()); }

static void sleep() { std::this_thread::sleep_for(std::chrono::seconds(10)); }

int main(int argc, char **argv) {
  print_pid();
  puts("break here\n");
  sleep();
  return 0;
}
