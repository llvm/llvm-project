// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+predctrl < %s 2>&1| FileCheck %s

cfp rctx
dvp rctx
cpp rctx

// CHECK: specified cfp op requires a register
// CHECK: specified dvp op requires a register
// CHECK: specified cpp op requires a register

cfp x0, x1
dvp x1, x2
cpp x2, x3

// CHECK:      invalid operand for prediction restriction instruction
// CHECK-NEXT: cfp
// CHECK:      invalid operand for prediction restriction instruction
// CHECK-NEXT: dvp
// CHECK:      invalid operand for prediction restriction instruction
// CHECK-NEXT: cpp
