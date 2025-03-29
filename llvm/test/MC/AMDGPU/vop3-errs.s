// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck %s --check-prefix=GFX67 --check-prefix=GCN --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=fiji %s 2>&1 | FileCheck %s --check-prefix=GFX89 --check-prefix=GCN --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck %s --check-prefix=GFX89 --check-prefix=GCN --implicit-check-not=error:

v_add_f32_e64 v0, v1
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction

v_div_scale_f32  v24, vcc, v22, 1.1, v22
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

v_mqsad_u32_u8 v[0:3], s[2:3], v4, v[0:3]
// GFX67: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_eq_f32_e64 vcc, v0, v1 mul:2
// GFX67: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

v_cmp_le_f64_e64 vcc, v0, v1 mul:4
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

//
// mul
//

v_cvt_f64_i32 v[5:6], s1 mul:3
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid mul value.

//
// v_interp*
//

v_interp_mov_f32_e64 v5, p10, attr0.x high
// GFX67: :[[@LINE-1]]:{{[0-9]+}}: error: e64 variant of this instruction is not supported
// GFX89: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_mov_f32_e64 v5, p10, attr0.x v0
// GFX67: :[[@LINE-1]]:{{[0-9]+}}: error: e64 variant of this instruction is not supported
// GFX89: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p1_f32_e64 v5, v2, attr0.x high
// GFX67: :[[@LINE-1]]:{{[0-9]+}}: error: e64 variant of this instruction is not supported
// GFX89: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p1_f32_e64 v5, v2, attr0.x v0
// GFX67: :[[@LINE-1]]:{{[0-9]+}}: error: e64 variant of this instruction is not supported
// GFX89: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p2_f32_e64 v255, v2, attr0.x high
// GFX67: :[[@LINE-1]]:{{[0-9]+}}: error: e64 variant of this instruction is not supported
// GFX89: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p2_f32_e64 v255, v2, attr0.x v0
// GFX67: :[[@LINE-1]]:{{[0-9]+}}: error: e64 variant of this instruction is not supported
// GFX89: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p1ll_f16 v5, p0, attr31.x
// GFX67: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p1ll_f16 v5, v2, attr31.x v0
// GFX67: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p2_f16 v5, v2, attr1.x, v3 mul:2
// GFX67: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

//
// v_div_scale_*
//

v_div_scale_f32  v24, vcc, v22, v22, |v20|
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: ABS not allowed in VOP3B instructions

v_div_scale_f64  v[24:25], vcc, -|v[22:23]|, v[22:23], v[20:21]
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: ABS not allowed in VOP3B instructions
