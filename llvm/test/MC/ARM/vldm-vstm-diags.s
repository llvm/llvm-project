@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null             %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-D32
@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null -mattr=+d16 %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-D16

  // First operand must be a GPR
  vldm s0, {s1, s2}
// CHECK: error: operand must be a register in range [r0, r15]
// CHECK-NEXT: vldm s0, {s1, s2}

  vstm s0, {s1, s2}
// CHECK: error: operand must be a register in range [r0, r15]
// CHECK-NEXT: vstm s0, {s1, s2}


  // Second operand must be a list of SPRs or DPRs
  vldm r0, {r1, r2}
// CHECK: error: invalid instruction, any one of the following would fix this:
// CHECK-NEXT: vldm r0, {r1, r2}
// CHECK: note: operand must be a list of registers in range [s0, s31]
// CHECK-D32: note: operand must be a list of registers in range [d0, d31]
// CHECK-D16: note: operand must be a list of registers in range [d0, d15]
  vldm r0, #42
// CHECK: error: invalid instruction, any one of the following would fix this:
// CHECK-NEXT: vldm r0, #42
// CHECK: note: operand must be a list of registers in range [s0, s31]
// CHECK-D32: note: operand must be a list of registers in range [d0, d31]
// CHECK-D16: note: operand must be a list of registers in range [d0, d15]
  vldm r0, {s1, d2}
// CHECK: error: invalid register in register list
// CHECK-NEXT: vldm r0, {s1, d2}
  vstm r0, {r1, r2}
// CHECK: error: invalid instruction, any one of the following would fix this:
// CHECK-NEXT: vstm r0, {r1, r2}
// CHECK: note: operand must be a list of registers in range [s0, s31]
// CHECK-D32: note: operand must be a list of registers in range [d0, d31]
// CHECK-D16: note: operand must be a list of registers in range [d0, d15]
  vstm r0, #42
// CHECK: error: invalid instruction, any one of the following would fix this:
// CHECK-NEXT: vstm r0, #42
// CHECK: note: operand must be a list of registers in range [s0, s31]
// CHECK-D32: note: operand must be a list of registers in range [d0, d31]
// CHECK-D16: note: operand must be a list of registers in range [d0, d15]
  vstm r0, {s1, d2}
// CHECK: error: invalid register in register list
// CHECK-NEXT: vstm r0, {s1, d2}
