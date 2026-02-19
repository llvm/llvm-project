#include <sys/wait.h>
#include <unistd.h>

int fork_and_return(int value) {
  pid_t pid = fork();
  if (pid == -1)
    return -1;
  if (pid == 0) {
    // child
    _exit(0);
  }
  // parent
  int status;
  waitpid(pid, &status, 0);
  return value;
}

int main() {
  int x = 42;
  return 0; // break here
}
