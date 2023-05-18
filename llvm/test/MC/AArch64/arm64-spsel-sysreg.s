// RUN: not llvm-mc -triple arm64 -show-encoding < %s 2>%t | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERRORS < %t %s

msr SPSel, #0
msr SPSel, x0
msr DAIFSet, #0
msr ESR_EL1, x0
mrs x0, SPSel
mrs x0, ESR_EL1

// CHECK: msr SPSel, #0               // encoding: [0xbf,0x40,0x00,0xd5]
// CHECK: msr SPSel, x0               // encoding: [0x00,0x42,0x18,0xd5]
// CHECK: msr DAIFSet, #0             // encoding: [0xdf,0x40,0x03,0xd5]
// CHECK: msr ESR_EL1, x0             // encoding: [0x00,0x52,0x18,0xd5]
// CHECK: mrs x0, SPSel               // encoding: [0x00,0x42,0x38,0xd5]
// CHECK: mrs x0, ESR_EL1             // encoding: [0x00,0x52,0x38,0xd5]


msr DAIFSet, x0
msr ESR_EL1, #0
mrs x0, DAIFSet
// CHECK-ERRORS: error: immediate must be an integer in range [0, 15]
// CHECK-ERRORS: error: invalid operand for instruction
// CHECK-ERRORS: error: expected readable system register
