volatile long g;

__attribute__((noinline)) void callee(void) { g++; }

// Uses enough callee-saved registers that its prologue is worth outlining.
__attribute__((noinline)) long middle(long a, long b, long c, long d, long e,
                                      long f) {
  long sum = a + b * 2 + c * 3 + d * 4 + e * 5 + f * 6;
  callee();
  return sum + a - b + c - d + e - f;
}

// p, q, r and s stay live across the call, so they are held in the very
// callee-saved registers that middle()'s outlined prologue spills.
__attribute__((noinline)) long caller(long p, long q, long r, long s) {
  return middle(p, q, r, s, p + q, p - q) + p * 1000 + q * 100 + r * 10 + s;
}

int main(void) { return (int)caller(0x1111, 0x2222, 0x3333, 0x4444); }
