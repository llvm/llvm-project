void foo() {}

int main() {
  //% self.filecheck("c", "blraa.c")
  // CHECK: stop reason = EXC_BAD_ACCESS
  // CHECK-NEXT: Note: Possible pointer authentication failure detected.
  // CHECK-NEXT: Found authenticated indirect branch at address=0x{{.*}} (blraa.c:[[@LINE+1]]:3).
  asm volatile (
      "mov x9, #0xbad \n"
      "blraa %[target], x9 \n"
      /* Outputs */  :
      /* Inputs */   : [target] "r"(&foo)
      /* Clobbers */ : "x9"
  );

  return 1;
}

// Expected codegen and exception message without ptrauth diagnostics:
// * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x2000000100007f9c)
//     frame #0: 0x0000000100007f9c blraa2`foo
// blraa2`foo:
//     0x100007f9c <+0>: ret
//
// blraa2`main:
//     0x100007fa0 <+0>: nop
//     0x100007fa4 <+4>: ldr    x8, #0x5c
//     0x100007fa8 <+8>: mov    x9, #0xbad
