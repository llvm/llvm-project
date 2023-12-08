void foo() {}

int main() {
  //% self.filecheck("c", "braa.c")
  // CHECK: stop reason = EXC_BAD_ACCESS
  //
  // TODO: We need call site info support for indirect calls to make this work.
  // CHECK-NOT: pointer authentication failure
  asm volatile (
      "mov x9, #0xbad \n"
      "braa %[target], x9 \n"
      /* Outputs */  :
      /* Inputs */   : [target] "r"(&foo)
      /* Clobbers */ : "x9"
  );

  return 1;
}

// Expected codegen and exception message without ptrauth diagnostics:
// * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x2000000100007f9c)
//     frame #0: 0x0000000100007f9c braa`foo
// braa`foo:
//     0x100007f9c <+0>: ret
//
// braa`main:
//     0x100007fa0 <+0>: nop
//     0x100007fa4 <+4>: ldr    x8, #0x5c
//     0x100007fa8 <+8>: mov    x9, #0xbad
