__attribute__((noinline))
int regs_int_params(int a, int b, int c, int d, int e, int f) {
  // Keep all params in regs; avoid spilling to the stack.
  // The compiler will usually keep a..f in the 6 integer-arg regs.
  asm volatile("" :: "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f));
  return a + b + c + d + e + f;
}
