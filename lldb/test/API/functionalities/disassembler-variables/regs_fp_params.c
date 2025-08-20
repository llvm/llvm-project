__attribute__((noinline))
double regs_fp_params(double a, double b, double c, double d, double e, double f) {
  asm volatile("" :: "x"(a), "x"(b), "x"(c), "x"(d), "x"(e), "x"(f));
  return a + b + c + d + e + f;
}