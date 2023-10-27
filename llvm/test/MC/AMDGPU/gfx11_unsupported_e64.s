// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

v_dot2c_f32_f16_e64 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64 variant of this instruction is not supported

v_swap_b32_e64 v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e64 variant of this instruction is not supported
