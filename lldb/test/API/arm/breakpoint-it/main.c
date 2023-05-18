int main() {
  int value;
  asm (
      "cmp %1, %2\n\t"
      "ite ne\n\t"
      ".thumb_func\n\t"
      "bkpt_true:\n\t"
      "movne %0, %1\n\t"
      ".thumb_func\n\t"
      "bkpt_false:\n\t"
      "moveq %0, %2\n\t"
      : "=r" (value) : "r"(42), "r"(47));
  return value;
}
