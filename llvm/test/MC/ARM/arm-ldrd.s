// RUN: not llvm-mc -triple arm-eabi -mattr=+v5te %s -o /dev/null 2>&1 | FileCheck %s
//
// rdar://14479793

ldrd r1, r2, [pc, #0]
ldrd r1, r2, [r3, #4]
ldrd r1, r2, [r3], #4
ldrd r1, r2, [r3, #4]!
ldrd r1, r2, [r3, -r4]!
ldrd r1, r2, [r3, r4]
ldrd r1, r2, [r3], r4
// CHECK: error: Rt must be even-numbered
// CHECK: error: Rt must be even-numbered
// CHECK: error: Rt must be even-numbered
// CHECK: error: Rt must be even-numbered
// CHECK: error: Rt must be even-numbered
// CHECK: error: Rt must be even-numbered
// CHECK: error: Rt must be even-numbered

ldrd r0, r3, [pc, #0]
ldrd r0, r3, [r4, #4]
ldrd r0, r3, [r4], #4
ldrd r0, r3, [r4, #4]!
ldrd r0, r3, [r4, -r5]!
ldrd r0, r3, [r4, r5]
ldrd r0, r3, [r4], r5
// CHECK: error: destination operands must be sequential
// CHECK: error: destination operands must be sequential
// CHECK: error: destination operands must be sequential
// CHECK: error: destination operands must be sequential
// CHECK: error: destination operands must be sequential
// CHECK: error: destination operands must be sequential
// CHECK: error: destination operands must be sequential

ldrd lr, pc, [pc, #0]
ldrd lr, pc, [r3, #4]
ldrd lr, pc, [r3], #4
ldrd lr, pc, [r3, #4]!
ldrd lr, pc, [r3, -r4]!
ldrd lr, pc, [r3, r4]
ldrd lr, pc, [r3], r4
// CHECK: error: Rt can't be R14
// CHECK: error: Rt can't be R14
// CHECK: error: Rt can't be R14
// CHECK: error: Rt can't be R14
// CHECK: error: Rt can't be R14
// CHECK: error: Rt can't be R14
// CHECK: error: Rt can't be R14

ldrd r0, r1, [r0], #4
ldrd r0, r1, [r1], #4
ldrd r0, r1, [r0, #4]!
ldrd r0, r1, [r1, #4]!
// CHECK: error: base register needs to be different from destination registers
// CHECK: error: base register needs to be different from destination registers
// CHECK: error: base register needs to be different from destination registers
// CHECK: error: base register needs to be different from destination registers
