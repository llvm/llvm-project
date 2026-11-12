// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+specres2 < %s 2>&1| FileCheck %s

cosp rctx

// CHECK: specified cosp op requires a register

cosp x0, x1

// CHECK:      invalid operand for prediction restriction instruction
// CHECK-NEXT: cosp
