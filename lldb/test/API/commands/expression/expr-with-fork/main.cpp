#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

int fork_and_return(int value, bool use_vfork) {
  pid_t pid = use_vfork ? vfork() : fork();
  if (pid == -1)
    return -1;
  if (pid == 0) {
    // child
    _exit(value);
  }
  // parent
  int status;
  waitpid(pid, &status, 0);
  return WEXITSTATUS(status);
}

int fork_and_return_trap(int value) {
  pid_t pid = fork();
  if (pid == -1)
    return -1;
  if (pid == 0) {
    // child returning from the JITed function wrapper will hit a trap
    // instruction and terminate with SIGTRAP.
    return value;
  }
  // parent
  int status;
  waitpid(pid, &status, 0);
  if (WIFSIGNALED(status) && WTERMSIG(status) == SIGTRAP) {
    return 1; // Success: child terminated with SIGTRAP
  }
  return 0; // Failure
}

int main() {
  int x = 42;
  return 0; // break here
}
