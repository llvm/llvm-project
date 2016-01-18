//RUN: not llvm-mc -triple=aarch64-linux-gnu - < %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s

// simple test
.section a, "ax", @progbits
f1:
  ldr w0, =0x100000001
// CHECK-ERROR: error: Immediate too large for register
// CHECK-ERROR:   ldr w0, =0x100000001
// CHECK-ERROR:           ^
f2:
  ldr w0, =-0x80000001
// CHECK-ERROR: error: Immediate too large for register
// CHECK-ERROR:  ldr w0, =-0x80000001
// CHECK-ERROR:          ^
