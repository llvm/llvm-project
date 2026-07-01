// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 | FileCheck %s

stshh
stshh foo
stshh #0
stshh 0

// CHECK:      error: too few operands for instruction
// CHECK-NEXT: stshh
// CHECK-NEXT: ^~~~~
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: stshh foo
// CHECK-NEXT:       ^
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: stshh #0
// CHECK-NEXT:       ^
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: stshh 0
// CHECK-NEXT:       ^
