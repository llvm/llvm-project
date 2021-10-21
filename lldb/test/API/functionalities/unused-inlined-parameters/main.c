#include <stdio.h>

__attribute__((optnone)) __attribute__((nodebug)) void use(int used) {}

__attribute__((always_inline)) void f(void *unused1, int used, int unused2) {
  use(used); // break here
}

int main(int argc, char **argv) {
  f(argv, 42, 1);
  return 0;
}