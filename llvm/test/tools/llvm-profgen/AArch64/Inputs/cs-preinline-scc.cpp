// Mutual recursion: evalC (caller, out of line) <-> arith (callee, inlined into
// evalC by the Pass1 compiler). Mirrors evalCompressed <-> evalBinaryArith.
__attribute__((noinline)) unsigned long evalC(unsigned long n);

static unsigned long arith(unsigned long n) {
  unsigned long x = n * 7 + 3;
  x ^= x >> 5;
  x *= 31;
  if (n > 1)
    x += evalC(n - 2);
  return x;
}

__attribute__((noinline)) unsigned long evalC(unsigned long n) {
  unsigned long r = n + 1;
  if (n & 1)
    r += arith(n);
  return r;
}

int main(int argc, char **) { return (int)evalC(argc * 1000); }
