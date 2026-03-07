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

int main() {
  int x = 42;
  return 0; // break here
}
