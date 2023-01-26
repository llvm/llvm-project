// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

v_add_co_u32_e32 v2, vcc, s0, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e32 variant of this instruction is not supported

v_ashrrev_i16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e32 variant of this instruction is not supported

v_lshlrev_b16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e32 variant of this instruction is not supported

v_lshrrev_b16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e32 variant of this instruction is not supported

v_max_i16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e32 variant of this instruction is not supported

v_max_u16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e32 variant of this instruction is not supported

v_min_i16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e32 variant of this instruction is not supported

v_min_u16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e32 variant of this instruction is not supported

v_mul_lo_u16_e32 v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e32 variant of this instruction is not supported

v_sub_co_u32_e32 v2, vcc, s0, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e32 variant of this instruction is not supported

v_subrev_co_u32_e32 v2, vcc, s0, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: e32 variant of this instruction is not supported
