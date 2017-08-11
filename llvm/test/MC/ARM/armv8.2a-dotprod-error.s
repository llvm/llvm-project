// RUN: not llvm-mc -triple arm -mattr=+dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s
// RUN: not llvm-mc -triple thumb -mattr=+dotprod -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s

vudot.u8 d0, d1, d2[2]
vsdot.s8 d0, d1, d2[2]
vudot.u8 q0, q1, d4[2]
vsdot.s8 q0, q1, d4[2]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: error: invalid operand for instruction
