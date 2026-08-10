// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+rand < %s 2>&1| FileCheck %s

mrs rndr
mrs rndrrs

// CHECK:      invalid operand for instruction
// CHECK-NEXT: rndr
// CHECK:      invalid operand for instruction
// CHECK-NEXT: rndrrs

mrs rndr, x0
mrs rndrrs, x1

// CHECK:      invalid operand for instruction
// CHECK-NEXT: rndr
// CHECK:      invalid operand for instruction
// CHECK-NEXT: rndrrs
