#include <assert.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
  // break here
  pid_t pid = fork();
  if (pid == 0) {
    // child
    _exit(47);
  }
  // parent
  _exit(0);
}
