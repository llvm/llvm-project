#include <stdio.h>

// For the test case, we really want the the layout of this binary
// to be:
//
//   foo()
//   bar() - 4096 bytes of nop's
//   main()
//   "HI" string
//
// in reality getting this layout from the compiler and linker
// is a crapshoot, so I have yaml's checked in of the correct
// layout.  Recompiling from source may not get the needed
// binary layout.

static int bar();
static int foo() { return 5 + bar(); }
// A function of 4096 bytes, so when main() loads the
// address of foo() before this one, it has to subtract
// a 4096 page.
#define SIXTY_FOUR_BYTES_NOP                                                   \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");                                                                  \
  asm("nop");

static int bar() {
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  SIXTY_FOUR_BYTES_NOP;
  return 5;
}
int main() {
  int (*f)(void) = foo;
  puts("HI");
  return f();
}
