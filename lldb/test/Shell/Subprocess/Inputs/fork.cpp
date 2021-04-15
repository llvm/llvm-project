#include <sys/types.h>
#include <sys/wait.h>
#include <assert.h>
#if defined(TEST_CLONE)
#include <sched.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

int g_val = 0;

void parent_func() {
  g_val = 1;
  printf("function run in parent\n");
}

int child_func(void *unused) {
  // we need to avoid memory modifications for vfork(), yet we want
  // to be able to test watchpoints, so do the next best thing
  // and restore the original value
  g_val = 2;
  g_val = 0;
  return 0;
}

int main() {
  alignas(uintmax_t) char stack[4096];

#if defined(TEST_CLONE)
  pid_t pid = clone(child_func, &stack[sizeof(stack)], 0, NULL);
#elif defined(TEST_FORK)
  pid_t pid = TEST_FORK();
  if (pid == 0)
    _exit(child_func(NULL));
#endif
  assert(pid != -1);

  parent_func();
  int status, wait_flags = 0;
#if defined(TEST_CLONE)
  wait_flags = __WALL;
#endif
  pid_t waited = waitpid(pid, &status, wait_flags);
  assert(waited == pid);
  assert(WIFEXITED(status));
  printf("child exited: %d\n", WEXITSTATUS(status));

  return 0;
}
