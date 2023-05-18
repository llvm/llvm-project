// RUN: not llvm-mc -triple arm64 -mattr=neon %s 2> %t > /dev/null
// RUN: FileCheck %s < %t

        sqrdmulh v0.8h, v1.8h, v16.h[0]
// CHECK: error: invalid operand for instruction

        sqrdmulh h0, h1, v16.h[0]
// CHECK: error: invalid operand for instruction

        sqdmull2 v0.4h, v1.8h, v16.h[0]
// CHECK: error: invalid operand for instruction
