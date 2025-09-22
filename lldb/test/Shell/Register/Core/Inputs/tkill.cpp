#include <signal.h>
#include <sys/syscall.h>

int main() {
  // Get the current thread ID
  pid_t tid = syscall(SYS_gettid);
  // Send a SIGSEGV signal to the current thread
  syscall(SYS_tkill, tid, SIGSEGV);
  return 0;
}
