// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+bti < %s 2>&1 | FileCheck %s

bti cj
bti a
bti x0

// CHECK: invalid operand for instruction
// CHECK-NEXT: cj
// CHECK: invalid operand for instruction
// CHECK-NEXT: a
// CHECK: invalid operand for instruction
// CHECK-NEXT: x0
