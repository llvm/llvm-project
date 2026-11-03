#include <signal.h>
#include <string.h>
#include <unistd.h>

volatile int g_code;
void *volatile g_addr;

static void handler(int signo, siginfo_t *info, void *context) {
  g_code = info->si_code;
  g_addr = info->si_addr;
  _exit(0); // break in handler
}

void crash_at(volatile int *addr) { *addr = 47; }

int main() {
  struct sigaction action;
  memset(&action, 0, sizeof(action));
  action.sa_sigaction = handler;
  action.sa_flags = SA_SIGINFO;
  sigaction(SIGSEGV, &action, NULL);

  crash_at((volatile int *)0x1234);
  return 1;
}
