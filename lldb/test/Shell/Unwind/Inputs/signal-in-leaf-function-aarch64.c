#include <signal.h>
#include <unistd.h>

int __attribute__((naked)) signal_generating_add(int a, int b) {
  asm("add w0, w1, w0\n\t"
      "udf #0xdead\n\t"
      "ret");
}

void sigill_handler(int signo) { _exit(0); }

int main() {
  signal(SIGILL, sigill_handler);
  return signal_generating_add(42, 47);
}
