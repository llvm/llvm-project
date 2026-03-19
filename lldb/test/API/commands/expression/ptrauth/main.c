#include <stdio.h>

int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }

// Accepts a function pointer and calls it, preventing the compiler from
// folding an indirect call into a direct one in the expression evaluator.
int caller(int (*fn)(int, int), int a, int b) { return fn(a, b); }

// A global signed function pointer for testing auth of pointers from
// debuggee memory. Signed with IB key (__ptrauth(1, 0, 0)) which is
// process-specific, so this catches inter-process key mismatches.
int (*__ptrauth(1, 0, 0) global_fp)(int, int) = &add;

int main(void) {
  printf("%d %d\n", add(2, 3), mul(4, 5));
  return 0; // break here
}
