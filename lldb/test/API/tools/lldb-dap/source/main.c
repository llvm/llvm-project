#include <stdio.h>

__attribute__((nodebug)) static void add(int i, int j, void handler(int)) {
  handler(i + j);
}

static void handler(int result) {
  printf("result %d\n", result); // breakpoint
}

int main(int argc, char const *argv[]) {
  add(2, 3, handler);
  return 0;
}
