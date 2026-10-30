// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 | FileCheck %s

srls x0
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: srls x0
// CHECK-NEXT:      ^

srls inj
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: srls inj
// CHECK-NEXT:      ^

srls #0
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: srls #0
// CHECK-NEXT:      ^

srls 0
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: srls 0
// CHECK-NEXT:      ^

srls stshstrm, x0
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: srls stshstrm, x0
// CHECK-NEXT:                ^

slbnd x0
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: slbnd x0
// CHECK-NEXT:       ^
