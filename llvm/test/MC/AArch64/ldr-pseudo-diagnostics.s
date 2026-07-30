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

f3:
  ldr foo, =1
// CHECK-ERROR: error: Only valid when first operand is register
// CHECK-ERROR:   ldr foo, =1
// CHECK-ERROR:            ^

f4:
  add r0, r0, =1
// CHECK-ERROR: error: unexpected token in operand
// CHECK-ERROR:   add r0, r0, =1
// CHECK-ERROR:               ^

f5:
  ldr x0, =())
// CHECK-ERROR: error: unknown token in expression
// CHECK-ERROR:   ldr x0, =())
// CHECK-ERROR:             ^
