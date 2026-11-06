// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+fpv2 2>&1 < %s | FileCheck %s

fcmtl h0, s1
// CHECK: [[@LINE-1]]:11: error: invalid operand for instruction
// CHECK-NEXT: fcmtl h0, s1
// CHECK-NEXT:           ^

fcmtl s0, d1
// CHECK: [[@LINE-1]]:11: error: invalid operand for instruction
// CHECK-NEXT: fcmtl s0, d1
// CHECK-NEXT:           ^

fcmtl d0, h1
// CHECK: [[@LINE-1]]:11: error: invalid operand for instruction
// CHECK-NEXT: fcmtl d0, h1
// CHECK-NEXT:           ^

fcmtl b0, b1
// CHECK: [[@LINE-1]]:7: error: invalid operand for instruction
// CHECK-NEXT: fcmtl b0, b1
// CHECK-NEXT:       ^

fcmtlmg h0, s1
// CHECK: [[@LINE-1]]:13: error: invalid operand for instruction
// CHECK-NEXT: fcmtlmg h0, s1
// CHECK-NEXT:             ^

fcmtlmg s0, d1
// CHECK: [[@LINE-1]]:13: error: invalid operand for instruction
// CHECK-NEXT: fcmtlmg s0, d1
// CHECK-NEXT:             ^

fcmtlmg d0, h1
// CHECK: [[@LINE-1]]:13: error: invalid operand for instruction
// CHECK-NEXT: fcmtlmg d0, h1
// CHECK-NEXT:             ^

fcmtlmg b0, b1
// CHECK: [[@LINE-1]]:9: error: invalid operand for instruction
// CHECK-NEXT: fcmtlmg b0, b1
// CHECK-NEXT:         ^
