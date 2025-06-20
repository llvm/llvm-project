#include <signal.h>
#include <stdint.h>
#include <unistd.h>

void sigbus_handler(int signo) { _exit(47); }

int target_function() { return 47; }

int main() {
  signal(SIGBUS, sigbus_handler);

  // Generate a SIGBUS by deliverately calling through an unaligned function
  // pointer.
  union {
    int (*t)();
    uintptr_t p;
  } u;
  u.t = target_function;
  u.p |= 1;
  return u.t();
}
