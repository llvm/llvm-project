void __attribute__((naked)) cas(int *a, int *b) {
  // This atomic sequence implements a copy-and-swap function. This test should
  // stop at the first instruction, and after step instruction, we should stop
  // at the end of the sequence (on the ret instruction).
  asm volatile("1:\n\t"
               "lr.w a2, (a0)\n\t"
               "and a5, a2, a4\n\t"
               "beq a5, a1, 2f\n\t"
               "xor a5, a2, a0\n\t"
               "and a5, a5, a4\n\t"
               "xor a5, a2, a5\n\t"
               "sc.w a5, a1, (a3)\n\t"
               "beqz a5, 1b\n\t"
               "nop\n\t"
               "2:\n\t"
               "ret\n\t");
}

int main() {
  int a = 4;
  int b = 2;
  cas(&a, &b);
}
