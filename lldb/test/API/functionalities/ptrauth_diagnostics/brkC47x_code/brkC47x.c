void foo() {}

int main() {
  //% self.filecheck("c", "brkC47x.c")
  // CHECK: stop reason = EXC_BAD_ACCESS
  // CHECK-NEXT: Note: Possible pointer authentication failure detected.
  // CHECK-NEXT: Found value that failed to authenticate at address=0x{{.*}} (brkC47x.c:1:13).
  asm volatile (
      "mov x16, %[target] \n"
      "brk 0xc470 \n"
      /* Outputs */  :
      /* Inputs */   : [target] "r"(&foo)
      /* Clobbers */ : "x16"
  );

  return 1;
}
