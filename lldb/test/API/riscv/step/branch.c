void __attribute__((naked)) branch_cas(int *a, int *b) {
  // Stop at the first instruction. The atomic sequence contains active forward
  // branch (bne a5, a1, 2f). After step instruction lldb should stop at the
  // branch's target address (ret instruction).
  asm volatile("1:\n\t"
               "lr.w a2, (a0)\n\t"
               "and a5, a2, a4\n\t"
               "bne a5, a1, 2f\n\t"
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
  branch_cas(&a, &b);
}
