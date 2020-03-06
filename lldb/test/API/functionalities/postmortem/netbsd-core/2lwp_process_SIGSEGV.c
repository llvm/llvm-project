#include <lwp.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

volatile int sem = 0;

static void bar() {
  char F = 'b';
  sem = 1;
  while (1) continue; // Frame bar
}

static void foo(void (*boomer)()) {
  char F = 'f';
  boomer(); // Frame foo
}

static void lwp_main(void *unused) {
  char F = 'l';
  foo(bar); // Frame lwp_main
}

int main(int argc, char **argv) {
  ucontext_t uc;
  lwpid_t lid;
  static const size_t ssize = 16 * 1024;
  void *stack;

  stack = malloc(ssize);
  _lwp_makecontext(&uc, lwp_main, NULL, NULL, stack, ssize);
  _lwp_create(&uc, 0, &lid);
  while (sem != 1) continue;
  kill(getpid(), SIGSEGV);
  _lwp_wait(lid, NULL);
}
