// This program makes a multi tier nested function call to test AArch64
// Pointer Authentication feature.

// To enable PAC return address signing compile with following clang arguments:
// -march=armv8.3-a -mbranch-protection=pac-ret+leaf

#include <stdlib.h>

static void __attribute__((noinline)) func_c(void) {
  exit(0); // Frame func_c
}

static void __attribute__((noinline)) func_b(void) {
  func_c(); // Frame func_b
}

static void __attribute__((noinline)) func_a(void) {
  func_b(); // Frame func_a
}

int main(int argc, char *argv[]) {
  func_a(); // Frame main
  return 0;
}
