#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

void handler(int signo) {
  printf("SIGCHLD\n");
}

int main() {
  void *ret = signal(SIGINT, handler);
  assert (ret != SIG_ERR);

  pid_t child_pid = fork();
  assert (child_pid != -1);

  if (child_pid == 0) {
    sleep(1);
    _exit(14);
  }

  printf("signo = %d\n", SIGCHLD);
  printf("code = %d\n", CLD_EXITED);
  printf("child_pid = %d\n", child_pid);
  printf("uid = %d\n", getuid());
  pid_t waited = wait(NULL);
  assert(waited == child_pid);

  return 0;
}
