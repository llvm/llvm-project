// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx908 %s 2>&1 | FileCheck -check-prefix=GFX908 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefix=GFX900 %s

v_accvgpr_read_b32 v0, v0
// GFX908: error: invalid operand for instruction

v_accvgpr_read_b32 a0, a0
// GFX908: error: invalid operand for instruction

v_accvgpr_read_b32 v0, 1
// GFX908: error: invalid operand for instruction

v_accvgpr_read_b32 v0, s0
// GFX908: error: invalid operand for instruction

v_accvgpr_read_b32 v0, a0
// GFX900: error: instruction not supported on this GPU

v_accvgpr_write_b32 v0, v0
// GFX908: error: invalid operand for instruction

v_accvgpr_write_b32 a0, a0
// GFX908: error: invalid operand for instruction

v_accvgpr_write_b32 a0, s0
// GFX908: error: invalid operand for instruction

v_accvgpr_write_b32 a0, 65
// GFX908: error: invalid operand for instruction

v_accvgpr_write_b32 a0, v0
// GFX900: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 v[0:31], v0, v1, a[1:32]
// GFX908: error: not a valid operand

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, v[1:32]
// GFX908: error: not a valid operand

v_mfma_f32_32x32x1f32 a[0:31], s0, v1, a[1:32]
// GFX908: error: invalid operand for instruction

v_mfma_f32_32x32x1f32 a[0:31], 1, v1, a[1:32]
// GFX908: error: invalid operand for instruction

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, 65
// GFX908: error: invalid operand for instruction

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, 0
// GFX900: error: instruction not supported on this GPU
