// The breakpoint is in an inlined function, so frame #0 is an inlined frame.

volatile int g_value;

static inline __attribute__((always_inline)) void inlined(void) {
  ++g_value; // Break here.
}

__attribute__((noinline)) void callee(void) { inlined(); }

int main(void) {
  callee();
  return 0;
}
