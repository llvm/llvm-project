#include <stdio.h>

int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }

// A global signed function pointer for testing auth of pointers from
// debuggee memory.
int (*global_fp)(int, int) = &add;

int main(void) {
  printf("%d %d\n", add(2, 3), mul(4, 5));
  return 0; // break here
}
