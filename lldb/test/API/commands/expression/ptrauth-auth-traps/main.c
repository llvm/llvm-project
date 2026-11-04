#include <ptrauth.h>

int add(int a, int b) { return a + b; }

// Forces a genuine indirect call, preventing the compiler from folding the
// function pointer call into a direct call in the expression evaluator.
int caller(int (*fn)(int, int), int a, int b) { return fn(a, b); }

int main(void) {
  return 0; // break here
}
