// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck --check-prefix=GFX12 --implicit-check-not=error: %s

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 clamp
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 op_sel:[1,0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 op_sel:[0,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 op_sel:[0,0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 op_sel_hi:[0,1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 op_sel_hi:[1,0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 op_sel_hi:[1,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 neg_lo:[1,0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 neg_lo:[0,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 neg_hi:[1,0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 neg_hi:[0,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_dot4_f32_bf8_fp8 v0, v1, v2, v3 clamp
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_bf8_fp8 v0, v1, v2, v3 op_sel:[1,0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_bf8_fp8 v0, v1, v2, v3 op_sel:[0,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_bf8_fp8 v0, v1, v2, v3 op_sel:[0,0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_bf8_fp8 v0, v1, v2, v3 op_sel_hi:[0,1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_bf8_fp8 v0, v1, v2, v3 op_sel_hi:[1,0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_bf8_fp8 v0, v1, v2, v3 op_sel_hi:[1,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_bf8_fp8 v0, v1, v2, v3 neg_lo:[1,0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_dot4_f32_bf8_fp8 v0, v1, v2, v3 neg_lo:[0,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_dot4_f32_bf8_fp8 v0, v1, v2, v3 neg_hi:[1,0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_dot4_f32_bf8_fp8 v0, v1, v2, v3 neg_hi:[0,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_dot4_f32_fp8_fp8 v0, v1, v2, v3 clamp
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_fp8_fp8 v0, v1, v2, v3 op_sel:[1,0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_fp8_fp8 v0, v1, v2, v3 op_sel:[0,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_fp8_fp8 v0, v1, v2, v3 op_sel:[0,0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_fp8_fp8 v0, v1, v2, v3 op_sel_hi:[0,1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_fp8_fp8 v0, v1, v2, v3 op_sel_hi:[1,0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_fp8_fp8 v0, v1, v2, v3 op_sel_hi:[1,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_fp8_fp8 v0, v1, v2, v3 neg_lo:[1,0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_dot4_f32_fp8_fp8 v0, v1, v2, v3 neg_lo:[0,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_dot4_f32_fp8_fp8 v0, v1, v2, v3 neg_hi:[1,0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_dot4_f32_fp8_fp8 v0, v1, v2, v3 neg_hi:[0,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_dot4_f32_bf8_bf8 v0, v1, v2, v3 clamp
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_bf8_bf8 v0, v1, v2, v3 op_sel:[1,0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_bf8_bf8 v0, v1, v2, v3 op_sel:[0,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_bf8_bf8 v0, v1, v2, v3 op_sel:[0,0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_bf8_bf8 v0, v1, v2, v3 op_sel_hi:[0,1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_bf8_bf8 v0, v1, v2, v3 op_sel_hi:[1,0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_bf8_bf8 v0, v1, v2, v3 op_sel_hi:[1,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_dot4_f32_bf8_bf8 v0, v1, v2, v3 neg_lo:[1,0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_dot4_f32_bf8_bf8 v0, v1, v2, v3 neg_lo:[0,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_dot4_f32_bf8_bf8 v0, v1, v2, v3 neg_hi:[1,0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_dot4_f32_bf8_bf8 v0, v1, v2, v3 neg_hi:[0,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

// op_sel not allowed in dot opcodes with 4- or 8-bit packed data

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
