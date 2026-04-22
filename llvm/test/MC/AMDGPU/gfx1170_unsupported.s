// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1170 %s 2>&1 | FileCheck --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// Unsupported instructions.
//===----------------------------------------------------------------------===//

v_dot2c_f32_f16 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2acc_f32_f16 v5, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
