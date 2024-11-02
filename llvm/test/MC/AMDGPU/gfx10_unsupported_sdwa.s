// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1013 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

v_add_co_u32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_ashrrev_i16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_lshlrev_b16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_lshrrev_b16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_mac_f32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_max_i16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_max_u16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_min_i16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_min_u16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_mul_lo_u16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_sub_co_u32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_subrev_co_u32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
