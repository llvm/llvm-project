// RUN: not llvm-mc -triple arm -mattr=+dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s
// RUN: not llvm-mc -triple thumb -mattr=+dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s

// Only indices 0 an 1 should be accepted:

vudot.u8 d0, d1, d2[2]
vsdot.s8 d0, d1, d2[2]
vudot.u8 q0, q1, d4[2]
vsdot.s8 q0, q1, d4[2]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: vudot.u8 d0, d1, d2[2]
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: vsdot.s8 d0, d1, d2[2]
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: vudot.u8 q0, q1, d4[2]
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: vsdot.s8 q0, q1, d4[2]
// CHECK-ERROR:                    ^

// Only the lower 16 D-registers should be accepted:

vudot.u8 q0, q1, d16[0]
vsdot.s8 q0, q1, d16[0]

// CHECK-ERROR: error: operand must be a register in range [d0, d15]
// CHECK-ERROR: vudot.u8 q0, q1, d16[0]
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: operand must be a register in range [d0, d15]
// CHECK-ERROR: vsdot.s8 q0, q1, d16[0]
// CHECK-ERROR:                     ^
