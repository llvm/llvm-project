// RUN: not llvm-mc -triple armv6m--none-eabi < %s 2>&1 | FileCheck %s
  add sp, r0, #4
// CHECK:     error: invalid instruction, any one of the following would fix this:
// CHECK-NEXT:  add sp, r0, #4
// CHECK-NEXT:  ^
// CHECK-NEXT: note: operand must be a register in range [r0, r7]
// CHECK-NEXT:   add sp, r0, #4
// CHECK-NEXT:       ^
// CHECK-NEXT: note: operand must be a register sp
// CHECK-NEXT:  add sp, r0, #4
// CHECK-NEXT:          ^
// CHECK-NEXT: note: too many operands for instruction
// CHECK-NEXT:  add sp, r0, #4
// CHECK-NEXT:              ^
