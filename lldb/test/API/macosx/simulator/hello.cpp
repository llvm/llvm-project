#include <stdio.h>
#include <thread>
#if !defined(_WIN32)
#include <unistd.h>
#endif

static void print_pid() {
#if defined(_WIN32)
  fprintf(stderr, "PID: %d\n", ::GetCurrentProcessId());
#else
  fprintf(stderr, "PID: %d\n", getpid());
#endif
}

static void sleep() { std::this_thread::sleep_for(std::chrono::seconds(10)); }

int main(int argc, char **argv) {
  print_pid();
  puts("break here\n");
  sleep();
  return 0;
}
