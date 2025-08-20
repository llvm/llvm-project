__attribute__((noinline))
int loop_reg_rotate(int n, int seed) {
  volatile int acc = seed;    // keep as a named local
  int i = 0, j = 1, k = 2;    // extra pressure but not enough to spill

  for (int t = 0; t < n; ++t) {
    // Mix uses so the allocator may reshuffle regs for 'acc'
    acc = acc + i;
    asm volatile("" :: "r"(acc));      // pin 'acc' live here
    acc = acc ^ j;
    asm volatile("" :: "r"(acc));      // and here
    acc = acc + k;
    i ^= acc; j += acc; k ^= j;
  }

  asm volatile("" :: "r"(acc));
  return acc + i + j + k;
}
