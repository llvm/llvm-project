// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx908 %s 2>&1 | FileCheck -check-prefix=GFX908 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefix=GFX900 --implicit-check-not=error: %s

v_accvgpr_read_b32 v0, v0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_read_b32 a0, a0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_read_b32 v0, 1
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_read_b32 v0, s0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_read_b32 v0, a0
// GFX900: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_write_b32 v0, v0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_write_b32 a0, a0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_write_b32 a0, s0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: source operand must be either a VGPR or an inline constant
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_write_b32 a0, 65
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_accvgpr_write_b32 a0, v0
// GFX900: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 v[0:31], v0, v1, a[0:31]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, v[0:31]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], s0, v1, a[0:31]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], 1, v1, a[0:31]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, 65
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, 0
// GFX900: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX908: :[[@LINE-2]]:{{[0-9]+}}: error: inline constants are not allowed for this operand

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], v0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], a0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], a0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x1f32 a[0:15], v0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x1f32 a[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x1f32 a[0:15], v0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x1f32 a[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x1f32 a[0:15], a0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x1f32 a[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x1f32 a[0:15], a0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x1f32 a[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x1f32 a[0:3], v0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x1f32 a[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x1f32 a[0:3], v0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x1f32 a[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x1f32 a[0:3], a0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x1f32 a[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x1f32 a[0:3], a0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x1f32 a[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2f32 a[0:15], v0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2f32 a[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2f32 a[0:15], v0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2f32 a[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2f32 a[0:15], a0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2f32 a[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2f32 a[0:15], a0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2f32 a[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f32 a[0:3], v0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f32 a[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f32 a[0:3], v0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f32 a[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f32 a[0:3], a0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f32 a[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f32 a[0:3], a0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f32 a[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4f16 a[0:31], v[0:1], v[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4f16 a[0:31], v[0:1], v[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4f16 a[0:31], v[0:1], a[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4f16 a[0:31], v[0:1], a[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], v[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], v[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], a[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], a[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f16 a[0:15], v[0:1], v[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f16 a[0:15], v[0:1], v[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f16 a[0:15], v[0:1], a[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f16 a[0:15], v[0:1], a[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], v[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], v[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], a[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], a[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x4f16 a[0:3], v[0:1], v[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x4f16 a[0:3], v[0:1], v[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x4f16 a[0:3], v[0:1], a[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x4f16 a[0:3], v[0:1], a[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], v[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], v[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], a[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], a[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x8f16 a[0:15], v[0:1], v[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x8f16 a[0:15], v[0:1], a[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x8f16 a[0:15], v[0:1], a[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], v[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], v[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x16f16 a[0:3], v[0:1], a[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x16f16 a[0:3], v[0:1], a[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], v[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], v[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], a[1:2], -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], a[1:2], -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x4i8 a[0:31], v0, v1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x4i8 a[0:31], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x4i8 a[0:31], v0, a1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x4i8 a[0:31], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x4i8 a[0:31], a0, v1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x4i8 a[0:31], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x4i8 a[0:31], a0, a1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x4i8 a[0:31], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x4i8 a[0:15], v0, v1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x4i8 a[0:15], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x4i8 a[0:15], v0, a1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x4i8 a[0:15], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x4i8 a[0:15], a0, v1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x4i8 a[0:15], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x4i8 a[0:15], a0, a1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x4i8 a[0:15], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_4x4x4i8 a[0:3], v0, v1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_4x4x4i8 a[0:3], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_4x4x4i8 a[0:3], v0, a1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_4x4x4i8 a[0:3], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_4x4x4i8 a[0:3], a0, v1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_4x4x4i8 a[0:3], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_4x4x4i8 a[0:3], a0, a1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_4x4x4i8 a[0:3], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x8i8 a[0:15], v0, v1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x8i8 a[0:15], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x8i8 a[0:15], v0, a1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x8i8 a[0:15], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x8i8 a[0:15], a0, v1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x8i8 a[0:15], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x8i8 a[0:15], a0, a1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_32x32x8i8 a[0:15], a0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x16i8 a[0:3], v0, v1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x16i8 a[0:3], v0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x16i8 a[0:3], v0, a1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x16i8 a[0:3], v0, a1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x16i8 a[0:3], a0, v1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x16i8 a[0:3], a0, v1, 2 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x16i8 a[0:3], a0, a1, 2
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2bf16 a[0:31], v0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2bf16 a[0:31], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2bf16 a[0:31], v0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2bf16 a[0:31], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2bf16 a[0:31], a0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2bf16 a[0:31], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2bf16 a[0:31], a0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x2bf16 a[0:31], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x2bf16 a[0:15], v0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x2bf16 a[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x2bf16 a[0:15], v0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x2bf16 a[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x2bf16 a[0:15], a0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x2bf16 a[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x2bf16 a[0:15], a0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x2bf16 a[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x2bf16 a[0:3], v0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x2bf16 a[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x2bf16 a[0:3], v0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x2bf16 a[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x2bf16 a[0:3], a0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x2bf16 a[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x2bf16 a[0:3], a0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x2bf16 a[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4bf16 a[0:15], v0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4bf16 a[0:15], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4bf16 a[0:15], v0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4bf16 a[0:15], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4bf16 a[0:15], a0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4bf16 a[0:15], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4bf16 a[0:15], a0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4bf16 a[0:15], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8bf16 a[0:3], v0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8bf16 a[0:3], v0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8bf16 a[0:3], v0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8bf16 a[0:3], v0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8bf16 a[0:3], a0, v1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8bf16 a[0:3], a0, v1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8bf16 a[0:3], a0, a1, -2.0
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8bf16 a[0:3], a0, a1, -2.0 cbsz:3 abid:2 blgp:7
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: inline constants are not allowed for this operand
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, a[2:33]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: source 2 operand must not partially overlap with dst
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x1f32 a[0:15], v0, v1, a[2:17]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: source 2 operand must not partially overlap with dst
// GFX900: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
