// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=error: %s

// op_sel not allowed in dot opcodes with 4- or 8-bit packed data

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_iu8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
