#include <sys/types.h>
#include <sys/wait.h>
#include <assert.h>
#if defined(TEST_CLONE)
#include <sched.h>
#endif
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

int g_val = 0;

void parent_func() {
  g_val = 1;
  printf("function run in parent\n");
}

int parent_done[2];
char parent_done_str[16];

void wait_for_parent() {
  char buf[2];
  // wait for the parent to finish its part
  int ret = read(parent_done[0], buf, sizeof(buf));
  assert(ret == 2);
  ret = close(parent_done[0]);
  assert(ret == 0);
}

// This is the function we set breakpoint on.
int child_func(const char *argv0) {
  // we need to avoid memory modifications for vfork(), yet we want
  // to be able to test watchpoints, so do the next best thing
  // and restore the original value
  g_val = 2;
  g_val = 0;
  execl(argv0, argv0, parent_done_str, NULL);
  assert(0 && "this should not be reached");
  return 1;
}

int child_top_func(void *argv0_ptr) {
  const char *argv0 = static_cast<char*>(argv0_ptr);

  int ret = close(parent_done[1]);
  assert(ret == 0);

  // NB: when using vfork(), the parent may be suspended while running
  // this function, so do not rely on any synchronization until we exec
#if defined(TEST_FORK)
  if (TEST_FORK != vfork)
#endif
    wait_for_parent();

  return child_func(argv0);
}

int main(int argc, char* argv[]) {
  alignas(uintmax_t) char stack[4096];
  int ret;

  if (argv[1]) {
    parent_done[0] = atoi(argv[1]);
    assert(parent_done[0] != 0);

#if defined(TEST_FORK)
    // for vfork(), we need to synchronize after exec
    if (TEST_FORK == vfork)
      wait_for_parent();
#endif

    fprintf(stderr, "function run in exec'd child\n");
    return 0;
  }

  ret = pipe(parent_done);
  assert(ret == 0);

  ret = snprintf(parent_done_str, sizeof(parent_done_str),
                 "%d", parent_done[0]);
  assert(ret != -1);

#if defined(TEST_CLONE)
  pid_t pid = clone(child_top_func, &stack[sizeof(stack)], 0, argv[0]);
#elif defined(TEST_FORK)
  pid_t pid = TEST_FORK();
  // NB: this must be equivalent to the clone() call above
  if (pid == 0)
    _exit(child_top_func(argv[0]));
#endif
  assert(pid != -1);

  ret = close(parent_done[0]);
  assert(ret == 0);

  parent_func();

  // resume the child
  ret = write(parent_done[1], "go", 2);
  assert(ret == 2);
  ret = close(parent_done[1]);
  assert(ret == 0);

  int status, wait_flags = 0;
#if defined(TEST_CLONE)
  wait_flags = __WALL;
#endif
  pid_t waited = waitpid(pid, &status, wait_flags);
  assert(waited == pid);
  assert(WIFEXITED(status));
  assert(WEXITSTATUS(status) == 0);

  return 0;
}
