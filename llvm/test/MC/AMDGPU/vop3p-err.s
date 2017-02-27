// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx901 -show-encoding %s 2>&1 | FileCheck -check-prefix=GFX9 %s

// GFX9: 31: error: failed parsing operand.
v_pk_add_u16 v1, v2, v3 op_sel

// GFX9: 32: error: failed parsing operand.
v_pk_add_u16 v1, v2, v3 op_sel:

// GFX9: 33: error: failed parsing operand.
v_pk_add_u16 v1, v2, v3 op_sel:[

// GFX9: 33: error: failed parsing operand.
v_pk_add_u16 v1, v2, v3 op_sel:[]

// GFX9: 34: error: failed parsing operand.
v_pk_add_u16 v1, v2, v3 op_sel:[,]

// XXGFX9: 34: error: failed parsing operand.
// v_pk_add_u16 v1, v2, v3 op_sel:[0]

// GFX9: 35: error: failed parsing operand.
v_pk_add_u16 v1, v2, v3 op_sel:[0,]

// XXGFX9: 36: error: failed parsing operand.
// v_pk_add_u16 v1, v2, v3 op_sel:[,0]

// GFX9: 36: error: failed parsing operand.
v_pk_add_u16 v1, v2, v3 op_sel:[0,2]

// GFX9: 35: error: failed parsing operand.
v_pk_add_u16 v1, v2, v3 op_sel:[2,0]

// GFX9: 33: error: failed parsing operand.
v_pk_add_u16 v1, v2, v3 op_sel:[-1,0]

// GFX9: 35: error: failed parsing operand.
v_pk_add_u16 v1, v2, v3 op_sel:[0,-1]

// GFX9: 40: error: not a valid operand.
v_pk_add_u16 v1, v2, v3 op_sel:[0,0,0,0]

// XXGFX9: invalid operand for instruction
v_pk_add_u16 v1, v2, v3 neg_lo:[0,0]

//
// Regular modifiers on packed instructions
//

// FIXME: should be invalid operand for instruction
// GFX9: :18: error: not a valid operand.
v_pk_add_f16 v1, |v2|, v3

// GFX9: :21: error: not a valid operand.
v_pk_add_f16 v1, abs(v2), v3

// GFX9: :22: error: not a valid operand.
v_pk_add_f16 v1, v2, |v3|

// GFX9: :25: error: not a valid operand.
v_pk_add_f16 v1, v2, abs(v3)

// GFX9: :19: error: invalid operand for instruction
v_pk_add_f16 v1, -v2, v3

// GFX9: :23: error: invalid operand for instruction
v_pk_add_f16 v1, v2, -v3

// GFX9: :21: error: not a valid operand.
v_pk_add_u16 v1, abs(v2), v3

// GFX9: :19: error: invalid operand for instruction
v_pk_add_u16 v1, -v2, v3


//
// Packed operands on the non-packed VOP3P instructions
//

// GFX9: invalid operand for instruction
v_mad_mix_f32 v1, v2, v3, v4 op_sel:[0,0,0]

// GFX9: invalid operand for instruction
v_mad_mix_f32 v1, v2, v3, v4 op_sel_hi:[0,0,0]

// GFX9: invalid operand for instruction
v_mad_mix_f32 v1, v2, v3, v4 neg_lo:[0,0,0]

// GFX9: invalid operand for instruction
v_mad_mix_f32 v1, v2, v3, v4 neg_hi:[0,0,0]

// GFX9: invalid operand for instruction
v_mad_mixlo_f16 v1, v2, v3, v4 op_sel:[0,0,0]

// GFX9: invalid operand for instruction
v_mad_mixlo_f16 v1, v2, v3, v4 op_sel_hi:[0,0,0]

// GFX9: invalid operand for instruction
v_mad_mixlo_f16 v1, v2, v3, v4 neg_lo:[0,0,0]

// GFX9: invalid operand for instruction
v_mad_mixlo_f16 v1, v2, v3, v4 neg_hi:[0,0,0]

// GFX9: invalid operand for instruction
v_mad_mixhi_f16 v1, v2, v3, v4 op_sel:[0,0,0]

// GFX9: invalid operand for instruction
v_mad_mixhi_f16 v1, v2, v3, v4 op_sel_hi:[0,0,0]

// GFX9: invalid operand for instruction
v_mad_mixhi_f16 v1, v2, v3, v4 neg_lo:[0,0,0]

// GFX9: invalid operand for instruction
v_mad_mixhi_f16 v1, v2, v3, v4 neg_hi:[0,0,0]
