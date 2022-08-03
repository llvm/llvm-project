// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck -check-prefix=GFX1032 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck -check-prefix=GFX1064 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck -check-prefix=GFX1032-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck -check-prefix=GFX1064-ERR --implicit-check-not=error: %s

v_cmp_ge_i32_e32 s0, v0
// GFX1032: v_cmp_ge_i32_e32 vcc_lo, s0, v0 ; encoding: [0x00,0x00,0x0c,0x7d]
// GFX1064: v_cmp_ge_i32_e32 vcc, s0, v0 ; encoding: [0x00,0x00,0x0c,0x7d]

v_cmp_ge_i32_e32 vcc_lo, s0, v1
// GFX1032: v_cmp_ge_i32_e32 vcc_lo, s0, v1 ; encoding: [0x00,0x02,0x0c,0x7d]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_cmp_ge_i32_e32 vcc, s0, v2
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_cmp_ge_i32_e32 vcc, s0, v2 ; encoding: [0x00,0x04,0x0c,0x7d]

v_cmp_le_f16_sdwa s0, v3, v4 src0_sel:WORD_1 src1_sel:DWORD
// GFX1032: v_cmp_le_f16_sdwa s0, v3, v4 src0_sel:WORD_1 src1_sel:DWORD ; encoding: [0xf9,0x08,0x96,0x7d,0x03,0x80,0x05,0x06]
// GFX1064-ERR: :[[@LINE-2]]:19: error: invalid operand for instruction

v_cmp_le_f16_sdwa s[0:1], v3, v4 src0_sel:WORD_1 src1_sel:DWORD
// GFX1032-ERR: :[[@LINE-1]]:19: error: invalid operand for instruction
// GFX1064: v_cmp_le_f16_sdwa s[0:1], v3, v4 src0_sel:WORD_1 src1_sel:DWORD ; encoding: [0xf9,0x08,0x96,0x7d,0x03,0x80,0x05,0x06]

v_cmp_class_f32_e32 vcc_lo, s0, v0
// GFX1032: v_cmp_class_f32_e32 vcc_lo, s0, v0 ; encoding: [0x00,0x00,0x10,0x7d]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_cmp_class_f32_e32 vcc, s0, v0
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_cmp_class_f32_e32 vcc, s0, v0 ; encoding: [0x00,0x00,0x10,0x7d]

// TODO-GFX10: The following encoding does not match SP3's encoding, which is:
//  [0xf9,0x04,0x1e,0x7d,0x01,0x06,0x06,0x06]
v_cmp_class_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD
// GFX1032: v_cmp_class_f16_sdwa vcc_lo, v1, v2 src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x00,0x06,0x06]
// GFX1064-ERR: :[[@LINE-2]]:22: error: invalid operand for instruction

// TODO-GFX10: The following encoding does not match SP3's encoding, which is:
//  [0xf9,0x04,0x1e,0x7d,0x01,0x06,0x06,0x06]
v_cmp_class_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_cmp_class_f16_sdwa vcc, v1, v2 src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x00,0x06,0x06]

v_cmp_class_f16_sdwa s0, v1, v2 src0_sel:DWORD src1_sel:DWORD
// GFX1032: v_cmp_class_f16_sdwa s0, v1, v2 src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x80,0x06,0x06]
// GFX1064-ERR: :[[@LINE-2]]:22: error: invalid operand for instruction

v_cmp_class_f16_sdwa s[0:1], v1, v2 src0_sel:DWORD src1_sel:DWORD
// GFX1032-ERR: :[[@LINE-1]]:22: error: invalid operand for instruction
// GFX1064: v_cmp_class_f16_sdwa s[0:1], v1, v2 src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x1e,0x7d,0x01,0x80,0x06,0x06]

v_cndmask_b32_e32 v1, v2, v3,
// GFX1032: v_cndmask_b32_e32 v1, v2, v3, vcc_lo ; encoding: [0x02,0x07,0x02,0x02]
// GFX1064: v_cndmask_b32_e32 v1, v2, v3, vcc ; encoding: [0x02,0x07,0x02,0x02]

v_cndmask_b32_e32 v1, v2, v3, vcc_lo
// GFX1032: v_cndmask_b32_e32 v1, v2, v3, vcc_lo ; encoding: [0x02,0x07,0x02,0x02]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_cndmask_b32_e32 v1, v2, v3, vcc
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_cndmask_b32_e32 v1, v2, v3, vcc ; encoding: [0x02,0x07,0x02,0x02]

v_cndmask_b32_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD
// GFX1032: v_cndmask_b32_sdwa v5, v1, v2, vcc_lo  dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x02,0x01,0x16,0x06,0x06]
// GFX1064: v_cndmask_b32_sdwa v5, v1, v2, vcc  dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x02,0x01,0x16,0x06,0x06]

v_cndmask_b32_sdwa v5, v1, v2, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD
// GFX1032: v_cndmask_b32_sdwa v5, v1, v2, vcc_lo  dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x02,0x01,0x16,0x06,0x06]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_cndmask_b32_sdwa v5, v1, v2, vcc dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_cndmask_b32_sdwa v5, v1, v2, vcc  dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x02,0x01,0x16,0x06,0x06]

v_cndmask_b32_dpp v5, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032: v_cndmask_b32_dpp v5, v1, v2, vcc_lo  quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x02,0x01,0xe4,0x00,0x00]
// GFX1064: v_cndmask_b32_dpp v5, v1, v2, vcc  quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x02,0x01,0xe4,0x00,0x00]

v_cndmask_b32_dpp v5, v1, v2, vcc_lo quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032: v_cndmask_b32_dpp v5, v1, v2, vcc_lo  quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x02,0x01,0xe4,0x00,0x00]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_cndmask_b32_dpp v5, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_cndmask_b32_dpp v5, v1, v2, vcc  quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x02,0x01,0xe4,0x00,0x00]

v_add_co_u32_e32 v2, vcc_lo, s0, v2
// GFX1032-ERR: :[[@LINE-1]]:1: error: e32 variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: e32 variant of this instruction is not supported

v_add_co_u32_e32 v2, vcc, s0, v2
// GFX1032-ERR: :[[@LINE-1]]:1: error: e32 variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: e32 variant of this instruction is not supported

v_add_co_ci_u32_e32 v3, vcc_lo, v3, v4, vcc_lo
// GFX1032: v_add_co_ci_u32_e32 v3, vcc_lo, v3, v4, vcc_lo ; encoding: [0x03,0x09,0x06,0x50]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_add_co_ci_u32_e32 v3, vcc, v3, v4, vcc
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_add_co_ci_u32_e32 v3, vcc, v3, v4, vcc ; encoding: [0x03,0x09,0x06,0x50]

v_add_co_ci_u32_e32 v3, v3, v4
// GFX1032: v_add_co_ci_u32_e32 v3, vcc_lo, v3, v4, vcc_lo ; encoding: [0x03,0x09,0x06,0x50]
// GFX1064: v_add_co_ci_u32_e32 v3, vcc, v3, v4, vcc ; encoding: [0x03,0x09,0x06,0x50]

v_sub_co_u32_e32 v2, vcc_lo, s0, v2
// GFX1032-ERR: :[[@LINE-1]]:1: error: e32 variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: e32 variant of this instruction is not supported

v_sub_co_u32_e32 v2, vcc, s0, v2
// GFX1032-ERR: :[[@LINE-1]]:1: error: e32 variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: e32 variant of this instruction is not supported

v_subrev_co_u32_e32 v2, vcc_lo, s0, v2
// GFX1032-ERR: :[[@LINE-1]]:1: error: e32 variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: e32 variant of this instruction is not supported

v_subrev_co_u32_e32 v2, vcc, s0, v2
// GFX1032-ERR: :[[@LINE-1]]:1: error: e32 variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: e32 variant of this instruction is not supported

v_sub_co_ci_u32_e32 v3, vcc_lo, v3, v4, vcc_lo
// GFX1032: v_sub_co_ci_u32_e32 v3, vcc_lo, v3, v4, vcc_lo ; encoding: [0x03,0x09,0x06,0x52]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_sub_co_ci_u32_e32 v3, vcc, v3, v4, vcc
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_sub_co_ci_u32_e32 v3, vcc, v3, v4, vcc ; encoding: [0x03,0x09,0x06,0x52]

v_sub_co_ci_u32_e32 v3, v3, v4
// GFX1032: v_sub_co_ci_u32_e32 v3, vcc_lo, v3, v4, vcc_lo ; encoding: [0x03,0x09,0x06,0x52]
// GFX1064: v_sub_co_ci_u32_e32 v3, vcc, v3, v4, vcc ; encoding: [0x03,0x09,0x06,0x52]

v_subrev_co_ci_u32_e32 v1, vcc_lo, 0, v1, vcc_lo
// GFX1032: v_subrev_co_ci_u32_e32 v1, vcc_lo, 0, v1, vcc_lo ; encoding: [0x80,0x02,0x02,0x54]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_subrev_co_ci_u32_e32 v1, vcc, 0, v1, vcc
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_subrev_co_ci_u32_e32 v1, vcc, 0, v1, vcc ; encoding: [0x80,0x02,0x02,0x54]

v_subrev_co_ci_u32_e32 v1, 0, v1
// GFX1032: v_subrev_co_ci_u32_e32 v1, vcc_lo, 0, v1, vcc_lo ; encoding: [0x80,0x02,0x02,0x54]
// GFX1064: v_subrev_co_ci_u32_e32 v1, vcc, 0, v1, vcc ; encoding: [0x80,0x02,0x02,0x54]

v_add_co_u32_sdwa v0, vcc_lo, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// GFX1032-ERR: :[[@LINE-1]]:1: error: sdwa variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: sdwa variant of this instruction is not supported

v_add_co_u32_sdwa v0, vcc, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// GFX1032-ERR: :[[@LINE-1]]:1: error: sdwa variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: sdwa variant of this instruction is not supported

v_add_co_u32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// GFX1032-ERR: :[[@LINE-1]]:1: error: sdwa variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: sdwa variant of this instruction is not supported

v_add_co_ci_u32_sdwa v1, vcc_lo, v1, v4, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// GFX1032: v_add_co_ci_u32_sdwa v1, vcc_lo, v1, v4, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x50,0x01,0x06,0x00,0x06]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_add_co_ci_u32_sdwa v1, vcc, v1, v4, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_add_co_ci_u32_sdwa v1, vcc, v1, v4, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x50,0x01,0x06,0x00,0x06]

v_add_co_ci_u32_sdwa v1, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// GFX1032: v_add_co_ci_u32_sdwa v1, vcc_lo, v1, v4, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x50,0x01,0x06,0x00,0x06]
// GFX1064: v_add_co_ci_u32_sdwa v1, vcc, v1, v4, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x50,0x01,0x06,0x00,0x06]

v_sub_co_u32_sdwa v0, vcc_lo, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// GFX1032-ERR: :[[@LINE-1]]:1: error: sdwa variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: sdwa variant of this instruction is not supported

v_sub_co_u32_sdwa v0, vcc, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// GFX1032-ERR: :[[@LINE-1]]:1: error: sdwa variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: sdwa variant of this instruction is not supported

v_sub_co_u32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// GFX1032-ERR: :[[@LINE-1]]:1: error: sdwa variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: sdwa variant of this instruction is not supported

v_subrev_co_u32_sdwa v0, vcc_lo, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// GFX1032-ERR: :[[@LINE-1]]:1: error: sdwa variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: sdwa variant of this instruction is not supported

v_subrev_co_u32_sdwa v0, vcc, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// GFX1032-ERR: :[[@LINE-1]]:1: error: sdwa variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: sdwa variant of this instruction is not supported

v_subrev_co_u32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// GFX1032-ERR: :[[@LINE-1]]:1: error: sdwa variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: sdwa variant of this instruction is not supported

v_sub_co_ci_u32_sdwa v1, vcc_lo, v1, v4, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// GFX1032: v_sub_co_ci_u32_sdwa v1, vcc_lo, v1, v4, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x52,0x01,0x06,0x00,0x06]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_sub_co_ci_u32_sdwa v1, vcc, v1, v4, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_sub_co_ci_u32_sdwa v1, vcc, v1, v4, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x52,0x01,0x06,0x00,0x06]

v_sub_co_ci_u32_sdwa v1, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// GFX1032: v_sub_co_ci_u32_sdwa v1, vcc_lo, v1, v4, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x52,0x01,0x06,0x00,0x06]
// GFX1064: v_sub_co_ci_u32_sdwa v1, vcc, v1, v4, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x52,0x01,0x06,0x00,0x06]

v_subrev_co_ci_u32_sdwa v1, vcc_lo, v1, v4, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// GFX1032: v_subrev_co_ci_u32_sdwa v1, vcc_lo, v1, v4, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x54,0x01,0x06,0x00,0x06]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_subrev_co_ci_u32_sdwa v1, vcc, v1, v4, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_subrev_co_ci_u32_sdwa v1, vcc, v1, v4, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x54,0x01,0x06,0x00,0x06]

v_subrev_co_ci_u32_sdwa v1, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// GFX1032: v_subrev_co_ci_u32_sdwa v1, vcc_lo, v1, v4, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x54,0x01,0x06,0x00,0x06]
// GFX1064: v_subrev_co_ci_u32_sdwa v1, vcc, v1, v4, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x54,0x01,0x06,0x00,0x06]

v_add_co_ci_u32 v1, sext(v1), sext(v4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// GFX1032: v_add_co_ci_u32_sdwa v1, vcc_lo, sext(v1), sext(v4), vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x50,0x01,0x06,0x08,0x0e]
// GFX1064: v_add_co_ci_u32_sdwa v1, vcc, sext(v1), sext(v4), vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x50,0x01,0x06,0x08,0x0e]

v_add_co_ci_u32_sdwa v1, vcc_lo, sext(v1), sext(v4), vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// GFX1032: v_add_co_ci_u32_sdwa v1, vcc_lo, sext(v1), sext(v4), vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x50,0x01,0x06,0x08,0x0e]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_add_co_ci_u32_sdwa v1, vcc, sext(v1), sext(v4), vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_add_co_ci_u32_sdwa v1, vcc, sext(v1), sext(v4), vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x08,0x02,0x50,0x01,0x06,0x08,0x0e]

v_add_co_u32_dpp v5, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032-ERR: :[[@LINE-1]]:1: error: dpp variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: dpp variant of this instruction is not supported

v_add_co_u32_dpp v5, vcc_lo, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032-ERR: :[[@LINE-1]]:1: error: dpp variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: dpp variant of this instruction is not supported

v_add_co_u32_dpp v5, vcc, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032-ERR: :[[@LINE-1]]:1: error: dpp variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: dpp variant of this instruction is not supported

v_add_co_ci_u32_dpp v5, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032: v_add_co_ci_u32_dpp v5, vcc_lo, v1, v2, vcc_lo quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x50,0x01,0xe4,0x00,0x00]
// GFX1064: v_add_co_ci_u32_dpp v5, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x50,0x01,0xe4,0x00,0x00]

v_add_co_ci_u32_dpp v5, vcc_lo, v1, v2, vcc_lo quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032: v_add_co_ci_u32_dpp v5, vcc_lo, v1, v2, vcc_lo quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x50,0x01,0xe4,0x00,0x00]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_add_co_ci_u32_dpp v5, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_add_co_ci_u32_dpp v5, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x50,0x01,0xe4,0x00,0x00]

v_sub_co_u32_dpp v5, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032-ERR: :[[@LINE-1]]:1: error: dpp variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: dpp variant of this instruction is not supported

v_sub_co_u32_dpp v5, vcc_lo, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032-ERR: :[[@LINE-1]]:1: error: dpp variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: dpp variant of this instruction is not supported

v_sub_co_u32_dpp v5, vcc, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032-ERR: :[[@LINE-1]]:1: error: dpp variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: dpp variant of this instruction is not supported

v_sub_co_ci_u32_dpp v5, vcc_lo, v1, v2, vcc_lo quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032: v_sub_co_ci_u32_dpp v5, vcc_lo, v1, v2, vcc_lo quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x52,0x01,0xe4,0x00,0x00]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_sub_co_ci_u32_dpp v5, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_sub_co_ci_u32_dpp v5, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x52,0x01,0xe4,0x00,0x00]

v_subrev_co_u32_dpp v5, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032-ERR: :[[@LINE-1]]:1: error: dpp variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: dpp variant of this instruction is not supported

v_subrev_co_u32_dpp v5, vcc_lo, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032-ERR: :[[@LINE-1]]:1: error: dpp variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: dpp variant of this instruction is not supported

v_subrev_co_u32_dpp v5, vcc, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032-ERR: :[[@LINE-1]]:1: error: dpp variant of this instruction is not supported
// GFX1064-ERR: :[[@LINE-2]]:1: error: dpp variant of this instruction is not supported

v_subrev_co_ci_u32_dpp v5, vcc_lo, v1, v2, vcc_lo quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032: v_subrev_co_ci_u32_dpp v5, vcc_lo, v1, v2, vcc_lo quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x54,0x01,0xe4,0x00,0x00]
// GFX1064-ERR: :[[@LINE-2]]:1: error: operands are not valid for this GPU or mode

v_subrev_co_ci_u32_dpp v5, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX1032-ERR: :[[@LINE-1]]:1: error: operands are not valid for this GPU or mode
// GFX1064: v_subrev_co_ci_u32_dpp v5, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 ; encoding: [0xfa,0x04,0x0a,0x54,0x01,0xe4,0x00,0x00]

v_add_co_u32 v0, s0, v0, v2
// GFX1032: v_add_co_u32 v0, s0, v0, v2 ; encoding: [0x00,0x00,0x0f,0xd7,0x00,0x05,0x02,0x00]
// GFX1064-ERR: :[[@LINE-2]]:26: error: invalid operand for instruction

v_add_co_u32_e64 v0, s0, v0, v2
// GFX1032: v_add_co_u32 v0, s0, v0, v2 ; encoding: [0x00,0x00,0x0f,0xd7,0x00,0x05,0x02,0x00]
// GFX1064-ERR: :[[@LINE-2]]:30: error: invalid operand for instruction

v_add_co_ci_u32_e64 v4, s0, v1, v5, s2
// GFX1032: v_add_co_ci_u32_e64 v4, s0, v1, v5, s2 ; encoding: [0x04,0x00,0x28,0xd5,0x01,0x0b,0x0a,0x00]
// GFX1064-ERR: :[[@LINE-2]]:25: error: invalid operand for instruction

v_sub_co_u32 v0, s0, v0, v2
// GFX1032: v_sub_co_u32 v0, s0, v0, v2 ; encoding: [0x00,0x00,0x10,0xd7,0x00,0x05,0x02,0x00]
// GFX1064-ERR: :[[@LINE-2]]:26: error: invalid operand for instruction

v_sub_co_u32_e64 v0, s0, v0, v2
// GFX1032: v_sub_co_u32 v0, s0, v0, v2 ; encoding: [0x00,0x00,0x10,0xd7,0x00,0x05,0x02,0x00]
// GFX1064-ERR: :[[@LINE-2]]:30: error: invalid operand for instruction

v_sub_co_ci_u32_e64 v4, s0, v1, v5, s2
// GFX1032: v_sub_co_ci_u32_e64 v4, s0, v1, v5, s2 ; encoding: [0x04,0x00,0x29,0xd5,0x01,0x0b,0x0a,0x00]
// GFX1064-ERR: :[[@LINE-2]]:25: error: invalid operand for instruction

v_subrev_co_u32 v0, s0, v0, v2
// GFX1032: v_subrev_co_u32 v0, s0, v0, v2 ; encoding: [0x00,0x00,0x19,0xd7,0x00,0x05,0x02,0x00]
// GFX1064-ERR: :[[@LINE-2]]:29: error: invalid operand for instruction

v_subrev_co_u32_e64 v0, s0, v0, v2
// GFX1032: v_subrev_co_u32 v0, s0, v0, v2 ; encoding: [0x00,0x00,0x19,0xd7,0x00,0x05,0x02,0x00]
// GFX1064-ERR: :[[@LINE-2]]:33: error: invalid operand for instruction

v_subrev_co_ci_u32_e64 v4, s0, v1, v5, s2
// GFX1032: v_subrev_co_ci_u32_e64 v4, s0, v1, v5, s2 ; encoding: [0x04,0x00,0x2a,0xd5,0x01,0x0b,0x0a,0x00]
// GFX1064-ERR: :[[@LINE-2]]:28: error: invalid operand for instruction

v_add_co_u32 v0, s[0:1], v0, v2
// GFX1032-ERR: :[[@LINE-1]]:18: error: invalid operand for instruction
// GFX1064: v_add_co_u32 v0, s[0:1], v0, v2 ; encoding: [0x00,0x00,0x0f,0xd7,0x00,0x05,0x02,0x00]

v_add_co_u32_e64 v0, s[0:1], v0, v2
// GFX1032-ERR: :[[@LINE-1]]:22: error: invalid operand for instruction
// GFX1064: v_add_co_u32 v0, s[0:1], v0, v2 ; encoding: [0x00,0x00,0x0f,0xd7,0x00,0x05,0x02,0x00]

v_add_co_ci_u32_e64 v4, s[0:1], v1, v5, s[2:3]
// GFX1032-ERR: :[[@LINE-1]]:25: error: invalid operand for instruction
// GFX1064: v_add_co_ci_u32_e64 v4, s[0:1], v1, v5, s[2:3] ; encoding: [0x04,0x00,0x28,0xd5,0x01,0x0b,0x0a,0x00]

v_sub_co_u32 v0, s[0:1], v0, v2
// GFX1032-ERR: :[[@LINE-1]]:18: error: invalid operand for instruction
// GFX1064: v_sub_co_u32 v0, s[0:1], v0, v2 ; encoding: [0x00,0x00,0x10,0xd7,0x00,0x05,0x02,0x00]

v_sub_co_u32_e64 v0, s[0:1], v0, v2
// GFX1032-ERR: :[[@LINE-1]]:22: error: invalid operand for instruction
// GFX1064: v_sub_co_u32 v0, s[0:1], v0, v2 ; encoding: [0x00,0x00,0x10,0xd7,0x00,0x05,0x02,0x00]

v_sub_co_ci_u32_e64 v4, s[0:1], v1, v5, s[2:3]
// GFX1032-ERR: :[[@LINE-1]]:25: error: invalid operand for instruction
// GFX1064: v_sub_co_ci_u32_e64 v4, s[0:1], v1, v5, s[2:3] ; encoding: [0x04,0x00,0x29,0xd5,0x01,0x0b,0x0a,0x00]

v_subrev_co_u32 v0, s[0:1], v0, v2
// GFX1032-ERR: :[[@LINE-1]]:21: error: invalid operand for instruction
// GFX1064: v_subrev_co_u32 v0, s[0:1], v0, v2 ; encoding: [0x00,0x00,0x19,0xd7,0x00,0x05,0x02,0x00]

v_subrev_co_u32_e64 v0, s[0:1], v0, v2
// GFX1032-ERR: :[[@LINE-1]]:25: error: invalid operand for instruction
// GFX1064: v_subrev_co_u32 v0, s[0:1], v0, v2 ; encoding: [0x00,0x00,0x19,0xd7,0x00,0x05,0x02,0x00]

v_subrev_co_ci_u32_e64 v4, s[0:1], v1, v5, s[2:3]
// GFX1032-ERR: :[[@LINE-1]]:28: error: invalid operand for instruction
// GFX1064: v_subrev_co_ci_u32_e64 v4, s[0:1], v1, v5, s[2:3] ; encoding: [0x04,0x00,0x2a,0xd5,0x01,0x0b,0x0a,0x00]

v_add_co_ci_u32_e64 v4, vcc_lo, v1, v5, s2
// GFX1032: v_add_co_ci_u32_e64 v4, vcc_lo, v1, v5, s2 ; encoding: [0x04,0x6a,0x28,0xd5,0x01,0x0b,0x0a,0x00]
// GFX1064-ERR: :[[@LINE-2]]:41: error: invalid operand for instruction

v_add_co_ci_u32_e64 v4, vcc, v1, v5, s[2:3]
// GFX1032-ERR: :[[@LINE-1]]:38: error: invalid operand for instruction
// GFX1064: v_add_co_ci_u32_e64 v4, vcc, v1, v5, s[2:3] ; encoding: [0x04,0x6a,0x28,0xd5,0x01,0x0b,0x0a,0x00]

v_add_co_ci_u32_e64 v4, s0, v1, v5, vcc_lo
// GFX1032: v_add_co_ci_u32_e64 v4, s0, v1, v5, vcc_lo ; encoding: [0x04,0x00,0x28,0xd5,0x01,0x0b,0xaa,0x01]
// GFX1064-ERR: :[[@LINE-2]]:25: error: invalid operand for instruction

v_add_co_ci_u32_e64 v4, s[0:1], v1, v5, vcc
// GFX1032-ERR: :[[@LINE-1]]:25: error: invalid operand for instruction
// GFX1064: v_add_co_ci_u32_e64 v4, s[0:1], v1, v5, vcc ; encoding: [0x04,0x00,0x28,0xd5,0x01,0x0b,0xaa,0x01]

v_div_scale_f32 v2, s2, v0, v0, v2
// GFX1032: v_div_scale_f32 v2, s2, v0, v0, v2 ; encoding: [0x02,0x02,0x6d,0xd5,0x00,0x01,0x0a,0x04]
// GFX1064-ERR: :[[@LINE-2]]:21: error: invalid operand for instruction

v_div_scale_f32 v2, s[2:3], v0, v0, v2
// GFX1032-ERR: :[[@LINE-1]]:21: error: invalid operand for instruction
// GFX1064: v_div_scale_f32 v2, s[2:3], v0, v0, v2 ; encoding: [0x02,0x02,0x6d,0xd5,0x00,0x01,0x0a,0x04]

v_div_scale_f64 v[2:3], s2, v[0:1], v[0:1], v[2:3]
// GFX1032: v_div_scale_f64 v[2:3], s2, v[0:1], v[0:1], v[2:3] ; encoding: [0x02,0x02,0x6e,0xd5,0x00,0x01,0x0a,0x04]
// GFX1064-ERR: :[[@LINE-2]]:25: error: invalid operand for instruction

v_div_scale_f64 v[2:3], s[2:3], v[0:1], v[0:1], v[2:3]
// GFX1032-ERR: :[[@LINE-1]]:25: error: invalid operand for instruction
// GFX1064: v_div_scale_f64 v[2:3], s[2:3], v[0:1], v[0:1], v[2:3] ; encoding: [0x02,0x02,0x6e,0xd5,0x00,0x01,0x0a,0x04]

v_mad_i64_i32 v[0:1], s6, v0, v1, v[2:3]
// GFX1032: v_mad_i64_i32 v[0:1], s6, v0, v1, v[2:3] ; encoding: [0x00,0x06,0x77,0xd5,0x00,0x03,0x0a,0x04]
// GFX1064-ERR: :[[@LINE-2]]:23: error: invalid operand for instruction

v_mad_i64_i32 v[0:1], s[6:7], v0, v1, v[2:3]
// GFX1032-ERR: :[[@LINE-1]]:23: error: invalid operand for instruction
// GFX1064: v_mad_i64_i32 v[0:1], s[6:7], v0, v1, v[2:3] ; encoding: [0x00,0x06,0x77,0xd5,0x00,0x03,0x0a,0x04]

v_mad_u64_u32 v[0:1], s6, v0, v1, v[2:3]
// GFX1032: v_mad_u64_u32 v[0:1], s6, v0, v1, v[2:3] ; encoding: [0x00,0x06,0x76,0xd5,0x00,0x03,0x0a,0x04]
// GFX1064-ERR: :[[@LINE-2]]:23: error: invalid operand for instruction

v_mad_u64_u32 v[0:1], s[6:7], v0, v1, v[2:3]
// GFX1032-ERR: :[[@LINE-1]]:23: error: invalid operand for instruction
// GFX1064: v_mad_u64_u32 v[0:1], s[6:7], v0, v1, v[2:3] ; encoding: [0x00,0x06,0x76,0xd5,0x00,0x03,0x0a,0x04]

v_cmpx_neq_f32_e32 v0, v1
// GFX1032: v_cmpx_neq_f32_e32 v0, v1 ; encoding: [0x00,0x03,0x3a,0x7c]
// GFX1064: v_cmpx_neq_f32_e32 v0, v1 ; encoding: [0x00,0x03,0x3a,0x7c]

v_cmpx_neq_f32_sdwa v0, v1 src0_sel:WORD_1 src1_sel:DWORD
// GFX1032: v_cmpx_neq_f32_sdwa v0, v1 src0_sel:WORD_1 src1_sel:DWORD ; encoding: [0xf9,0x02,0x3a,0x7c,0x00,0x00,0x05,0x06]
// GFX1064: v_cmpx_neq_f32_sdwa v0, v1 src0_sel:WORD_1 src1_sel:DWORD ; encoding: [0xf9,0x02,0x3a,0x7c,0x00,0x00,0x05,0x06]

v_cmpx_eq_u32_sdwa v0, 1 src0_sel:WORD_1 src1_sel:DWORD
// GFX1032: v_cmpx_eq_u32_sdwa v0, 1 src0_sel:WORD_1 src1_sel:DWORD ; encoding: [0xf9,0x02,0xa5,0x7d,0x00,0x00,0x05,0x86]
// GFX1064: v_cmpx_eq_u32_sdwa v0, 1 src0_sel:WORD_1 src1_sel:DWORD ; encoding: [0xf9,0x02,0xa5,0x7d,0x00,0x00,0x05,0x86]

v_cmpx_class_f32_e64 v0, 1
// GFX1032: v_cmpx_class_f32_e64 v0, 1 ; encoding: [0x7e,0x00,0x98,0xd4,0x00,0x03,0x01,0x00]
// GFX1064: v_cmpx_class_f32_e64 v0, 1 ; encoding: [0x7e,0x00,0x98,0xd4,0x00,0x03,0x01,0x00]

v_cmpx_class_f32_sdwa v0, 1 src0_sel:WORD_1 src1_sel:DWORD
// GFX1032: v_cmpx_class_f32_sdwa v0, 1 src0_sel:WORD_1 src1_sel:DWORD ; encoding: [0xf9,0x02,0x31,0x7d,0x00,0x00,0x05,0x86]
// GFX1064: v_cmpx_class_f32_sdwa v0, 1 src0_sel:WORD_1 src1_sel:DWORD ; encoding: [0xf9,0x02,0x31,0x7d,0x00,0x00,0x05,0x86]
