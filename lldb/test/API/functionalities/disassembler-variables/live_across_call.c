// Declare a real external call so the compiler must respect ABI clobbers.
extern int leaf(int) __attribute__((noinline));

__attribute__((noinline))
int live_across_call(int x) {
  volatile int a = x;                // a starts in a GPR (from arg)
  asm volatile("" :: "r"(a));        // keep 'a' live in a register
  int r = leaf(a);                   // 'a' is live across the call
  asm volatile("" :: "r"(a), "r"(r));// still live afterwards
  return a + r;
}
