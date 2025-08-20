__attribute__((noinline))
double regs_mixed_params(int a, int b, double x, double y, int c, double z) {
  // Keep everything live; avoid spills.
  asm volatile("" :: "r"(a), "r"(b), "x"(x), "x"(y), "r"(c), "x"(z));
  // Some mixing so values stay in regs long enough to annotate.
  double r = (double)(a + b + c) + x + y + z;
  asm volatile("" :: "x"(r), "r"(a), "x"(x));
  return r;
}
