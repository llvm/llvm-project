int main() {
  //% self.filecheck("c", "ldraa.c")
  // CHECK: EXC_BAD_ACCESS
  // CHECK-NEXT: Note: Possible pointer authentication failure detected.
  // CHECK-NEXT: Found authenticated load instruction at address=0x{{.*}} (ldraa.c:[[@LINE+3]]:3).
  long long foo = 0;

  asm volatile (
      "ldraa x9, [%[target]] \n"
      /* Outputs */  :
      /* Inputs */   : [target] "r"(&foo)
      /* Clobbers */ :
  );

  return 1;
}

// Expected codegen, register state, and exception message without ptrauth diagnostics:
// * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x2000016fdffc38)
//     frame #0: 0x0000000100007fa8 ldraa`main + 12
// ldraa`main:
// ->  0x100007fa8 <+12>: ldraa  x9, [x8]
//     0x100007fac <+16>: orr    w0, wzr, #0x1
//     0x100007fb0 <+20>: add    sp, sp, #0x10             ; =0x10
//     0x100007fb4 <+24>: ret
// Target 0: (ldraa) stopped.
// (lldb) p/x $x8
// (unsigned long) $0 = 0x000000016fdffc38
// (lldb) x/8 $x8
// 0x16fdffc38: 0x00000000 0x00000000 0x80254f30 0x00000001
// 0x16fdffc48: 0x00000000 0x00000000 0x00000000 0x00000000
