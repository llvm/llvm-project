#include <lwp.h>
#include <stddef.h>
#include <stdlib.h>

static void bar(char *boom) {
  char F = 'b';
  *boom = 47; // Frame bar
}

static void foo(char *boom, void (*boomer)(char *)) {
  char F = 'f';
  boomer(boom); // Frame foo
}

void lwp_main(void *unused) {
  char F = 'l';
  foo(0, bar); // Frame lwp_main
}

int main(int argc, char **argv) {
  ucontext_t uc;
  lwpid_t lid;
  static const size_t ssize = 16 * 1024;
  void *stack;

  stack = malloc(ssize);
  _lwp_makecontext(&uc, lwp_main, NULL, NULL, stack, ssize);
  _lwp_create(&uc, 0, &lid);
  _lwp_wait(lid, NULL);
}
