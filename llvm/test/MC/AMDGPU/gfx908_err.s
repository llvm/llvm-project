// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx908 %s 2>&1 | FileCheck --check-prefix=GFX908 --implicit-check-not=error: %s

// op_sel/op_sel_hi: in VOP3P dot, op_sel must be 0, op_sel_hi cannot appear

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX908: :[[@LINE-1]]:{{[0-9]+}}: error: invalid op_sel operand
