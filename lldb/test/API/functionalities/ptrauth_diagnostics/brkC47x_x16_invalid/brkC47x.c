int main() {
  //% self.filecheck("c", "brkC47x.c")
  // CHECK: stop reason = EXC_BAD_ACCESS
  // CHECK-NOT: Note: Possible pointer authentication failure detected.
  asm volatile (
      "mov x16, #0xbad \n"
      "brk 0xc470 \n"
      /* Outputs */  :
      /* Inputs */   :
      /* Clobbers */ : "x16"
  );

  return 1;
}
