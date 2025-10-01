void __attribute__((naked)) incomplete_cas(int *a, int *b) {
  // Stop at the first instruction (an lr without a corresponding sc), then make
  // a step instruction and ensure that execution stops at the next instruction
  // (and a5, a2, a4).
  asm volatile("1:\n\t"
               "lr.w a2, (a0)\n\t"
               "and a5, a2, a4\n\t"
               "beq a5, a1, 2f\n\t"
               "xor a5, a2, a0\n\t"
               "and a5, a5, a4\n\t"
               "xor a5, a2, a5\n\t"
               "bnez a5, 1b\n\t"
               "2:\n\t"
               "ret\n\t");
}

int main() {
  int a = 4;
  int b = 2;
  incomplete_cas(&a, &b);
}
