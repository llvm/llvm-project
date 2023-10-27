#include <stdint.h>

void foo(void) {}
typedef void (*FooPtr)(void);

int main() {
  FooPtr fnptr = foo;
  // Set top byte.
  fnptr = (FooPtr)((uintptr_t)fnptr | (uintptr_t)0xff << 56);
  // Then apply a PAuth signature to it.
  __asm__ __volatile__("pacdza %0" : "=r"(fnptr) : "r"(fnptr));
  // fnptr is now:
  // <8 bit top byte tag><pointer signature><virtual address>

  foo(); // Set break point at this line.

  return 0;
}
