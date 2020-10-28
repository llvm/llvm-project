// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx800 %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx906 %s 2>&1 | FileCheck %s --check-prefix=GFX906-GFX908
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx908 %s 2>&1 | FileCheck %s --check-prefix=GFX906-GFX908

//
// Test unsupported GPUs.
//

// CHECK: error: instruction not supported on this GPU
v_fmac_f32 v0, v1, v2
// CHECK: error: instruction not supported on this GPU
v_xnor_b32 v0, v1, v2
// CHECK: error: instruction not supported on this GPU
v_dot2_f32_f16 v0, v1, v2, v3
// CHECK: error: instruction not supported on this GPU
v_dot2_i32_i16 v0, v1, v2, v3
// CHECK: error: instruction not supported on this GPU
v_dot2_u32_u16 v0, v1, v2, v3
// CHECK: error: instruction not supported on this GPU
v_dot4_i32_i8 v0, v1, v2, v3
// CHECK: error: instruction not supported on this GPU
v_dot4_u32_u8 v0, v1, v2, v3
// CHECK: error: instruction not supported on this GPU
v_dot8_i32_i4 v0, v1, v2, v3
// CHECK: error: instruction not supported on this GPU
v_dot8_u32_u4 v0, v1, v2, v3

//
// Test invalid operands.
//

// GFX906-GFX908: error: invalid operand for instruction
v_dot2_f32_f16 v0, v1, v2, v3 op_sel
// GFX906-GFX908: error: expected a left square bracket
v_dot2_f32_f16 v0, v1, v2, v3 op_sel:
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[
// GFX906-GFX908: error: expected a left square bracket
v_dot2_f32_f16 v0, v1, v2, v3 op_sel:]
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[]
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[,]
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[,0]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,2]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[2,0]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[2,2]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,-1]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[-1,0]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[-1,-1]
// GFX906-GFX908: error: expected a closing square bracket
v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,0,0,0,0]
// GFX906-GFX908: error: invalid operand for instruction
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi
// GFX906-GFX908: error: expected a left square bracket
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[
// GFX906-GFX908: error: expected a left square bracket
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:]
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[]
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[,]
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[,0]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[0,2]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[2,0]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[2,2]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[0,-1]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[-1,0]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[-1,-1]
// GFX906-GFX908: error: expected a closing square bracket
v_dot2_f32_f16 v0, v1, v2, v3 op_sel_hi:[0,0,0,0,0]
// GFX906-GFX908: error: invalid operand for instruction
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo
// GFX906-GFX908: error: expected a left square bracket
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[
// GFX906-GFX908: error: expected a left square bracket
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:]
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[]
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[,]
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[,0]
// GFX906-GFX908: error: invalid neg_lo value.
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[0,2]
// GFX906-GFX908: error: invalid neg_lo value.
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[2,0]
// GFX906-GFX908: error: invalid neg_lo value.
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[2,2]
// GFX906-GFX908: error: invalid neg_lo value.
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[0,-1]
// GFX906-GFX908: error: invalid neg_lo value.
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[-1,0]
// GFX906-GFX908: error: invalid neg_lo value.
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[-1,-1]
// GFX906-GFX908: error: expected a closing square bracket
v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[0,0,0,0,0]
// GFX906-GFX908: error: invalid operand for instruction
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi
// GFX906-GFX908: error: expected a left square bracket
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[
// GFX906-GFX908: error: expected a left square bracket
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:]
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[]
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[,]
// GFX906-GFX908: error: unknown token in expression
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[,0]
// GFX906-GFX908: error: invalid neg_hi value.
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[0,2]
// GFX906-GFX908: error: invalid neg_hi value.
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[2,0]
// GFX906-GFX908: error: invalid neg_hi value.
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[2,2]
// GFX906-GFX908: error: invalid neg_hi value.
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[0,-1]
// GFX906-GFX908: error: invalid neg_hi value.
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[-1,0]
// GFX906-GFX908: error: invalid neg_hi value.
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[-1,-1]
// GFX906-GFX908: error: expected a closing square bracket
v_dot2_f32_f16 v0, v1, v2, v3 neg_hi:[0,0,0,0,0]
// GFX906-GFX908: error: invalid operand for instruction
v_dot2_i32_i16 v0, v1, v2, v3 op_sel
// GFX906-GFX908: error: expected a left square bracket
v_dot2_i32_i16 v0, v1, v2, v3 op_sel:
// GFX906-GFX908: error: unknown token in expression
v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[
// GFX906-GFX908: error: expected a left square bracket
v_dot2_i32_i16 v0, v1, v2, v3 op_sel:]
// GFX906-GFX908: error: unknown token in expression
v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[]
// GFX906-GFX908: error: unknown token in expression
v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[,]
// GFX906-GFX908: error: unknown token in expression
v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[,0]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,2]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[2,0]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[2,2]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,-1]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[-1,0]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[-1,-1]
// GFX906-GFX908: error: expected a closing square bracket
v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,0,0,0,0]
// GFX906-GFX908: error: invalid operand for instruction
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi
// GFX906-GFX908: error: expected a left square bracket
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:
// GFX906-GFX908: error: unknown token in expression
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[
// GFX906-GFX908: error: expected a left square bracket
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:]
// GFX906-GFX908: error: unknown token in expression
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[]
// GFX906-GFX908: error: unknown token in expression
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[,]
// GFX906-GFX908: error: unknown token in expression
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[,0]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[0,2]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[2,0]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[2,2]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[0,-1]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[-1,0]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[-1,-1]
// GFX906-GFX908: error: expected a closing square bracket
v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[0,0,0,0,0]
// FIXME-GFX906: error: invalid operand for instruction
v_dot2_i32_i16 v0, v1, v2, v3 neg_lo:[0,0]
// FIXME-GFX906: error: invalid operand for instruction
v_dot2_i32_i16 v0, v1, v2, v3 neg_hi:[0,0]
// GFX906-GFX908: error: invalid operand for instruction
v_dot2_u32_u16 v0, v1, v2, v3 op_sel
// GFX906-GFX908: error: expected a left square bracket
v_dot2_u32_u16 v0, v1, v2, v3 op_sel:
// GFX906-GFX908: error: unknown token in expression
v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[
// GFX906-GFX908: error: expected a left square bracket
v_dot2_u32_u16 v0, v1, v2, v3 op_sel:]
// GFX906-GFX908: error: unknown token in expression
v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[]
// GFX906-GFX908: error: unknown token in expression
v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[,]
// GFX906-GFX908: error: unknown token in expression
v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[,0]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[0,2]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[2,0]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[2,2]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[0,-1]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[-1,0]
// GFX906-GFX908: error: invalid op_sel value.
v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[-1,-1]
// GFX906-GFX908: error: expected a closing square bracket
v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[0,0,0,0,0]
// GFX906-GFX908: error: invalid operand for instruction
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi
// GFX906-GFX908: error: expected a left square bracket
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:
// GFX906-GFX908: error: unknown token in expression
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[
// GFX906-GFX908: error: expected a left square bracket
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:]
// GFX906-GFX908: error: unknown token in expression
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[]
// GFX906-GFX908: error: unknown token in expression
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[,]
// GFX906-GFX908: error: unknown token in expression
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[,0]
// GFX906-GFX908: error: invalid op_sel_hi value
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[0,2]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[2,0]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[2,2]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[0,-1]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[-1,0]
// GFX906-GFX908: error: invalid op_sel_hi value.
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[-1,-1]
// GFX906-GFX908: error: expected a closing square bracket
v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[0,0,0,0,0]
// FIXME-GFX906: error: invalid operand for instruction
v_dot2_u32_u16 v0, v1, v2, v3 neg_lo:[0,0]
// FIXME-GFX906: error: invalid operand for instruction
v_dot2_u32_u16 v0, v1, v2, v3 neg_hi:[0,0]

//
// Test regular modifiers.
//

// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, |v1|, v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, v1, |v2|, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, v1, v2, |v3|
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, |v1|, |v2|, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, |v1|, v2, |v3|
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, |v1|, |v2|, |v3|
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, abs(v1), v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, v1, abs(v2), v3
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, v1, v2, abs(v3)
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, abs(v1), abs(v2), v3
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, abs(v1), v2, abs(v3)
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, abs(v1), abs(v2), abs(v3)
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, -v1, v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, v1, -v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, v1, v2, -v3
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, -v1, -v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, -v1, v2, -v3
// GFX906-GFX908: error: not a valid operand
v_dot2_f32_f16 v0, -v1, -v2, -v3
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, |v1|, v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, v1, |v2|, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, v1, v2, |v3|
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, |v1|, |v2|, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, |v1|, v2, |v3|
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, |v1|, |v2|, |v3|
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, abs(v1), v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, v1, abs(v2), v3
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, v1, v2, abs(v3)
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, abs(v1), abs(v2), v3
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, abs(v1), v2, abs(v3)
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, abs(v1), abs(v2), abs(v3)
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, -v1, v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, v1, -v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, v1, v2, -v3
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, -v1, -v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, -v1, v2, -v3
// GFX906-GFX908: error: not a valid operand
v_dot2_i32_i16 v0, -v1, -v2, -v3
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, |v1|, v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, v1, |v2|, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, v1, v2, |v3|
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, |v1|, |v2|, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, |v1|, v2, |v3|
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, |v1|, |v2|, |v3|
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, abs(v1), v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, v1, abs(v2), v3
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, v1, v2, abs(v3)
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, abs(v1), abs(v2), v3
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, abs(v1), v2, abs(v3)
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, abs(v1), abs(v2), abs(v3)
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, -v1, v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, v1, -v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, v1, v2, -v3
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, -v1, -v2, v3
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, -v1, v2, -v3
// GFX906-GFX908: error: not a valid operand
v_dot2_u32_u16 v0, -v1, -v2, -v3

//
// Test constant bus restrictions.
//

// GFX906-GFX908: error: invalid operand (violates constant bus restrictions)
v_dot2_f32_f16 v255, s1, s2, s3
// GFX906-GFX908: error: invalid operand (violates constant bus restrictions)
v_dot2_i32_i16 v255, s1, s2, s3
// GFX906-GFX908: error: invalid operand (violates constant bus restrictions)
v_dot2_u32_u16 v255, s1, s2, s3
