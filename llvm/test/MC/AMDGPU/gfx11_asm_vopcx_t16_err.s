// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=error %s

v_cmpx_class_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_eq_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_eq_i16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_eq_u16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_f_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_ge_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_ge_i16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_ge_u16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_gt_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_gt_i16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_gt_u16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_le_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_le_i16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_le_u16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_lg_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_lt_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_lt_i16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_lt_u16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_ne_i16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_ne_u16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_neq_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_nge_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_ngt_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_nle_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_nlg_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_nlt_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_o_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_t_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_tru_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_u_f16_e32 v1, v255
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_class_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_eq_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_eq_i16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_eq_u16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_f_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_ge_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_ge_i16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_ge_u16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_gt_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_gt_i16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_gt_u16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_le_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_le_i16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_le_u16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_lg_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_lt_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_lt_i16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_lt_u16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_ne_i16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_ne_u16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_neq_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_nge_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_ngt_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_nle_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_nlg_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_nlt_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_o_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_t_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_tru_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_u_f16_e32 v255, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_cmpx_class_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_eq_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_eq_i16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_eq_u16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_f_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ge_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ge_i16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ge_u16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_gt_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_gt_i16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_gt_u16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_le_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_le_i16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_le_u16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lg_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lt_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lt_i16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lt_u16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ne_i16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ne_u16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_neq_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nge_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ngt_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nle_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nlg_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nlt_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_o_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_t_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_tru_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_u_f16_e32 v1, v255 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_class_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_eq_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_eq_i16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_eq_u16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_f_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ge_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ge_i16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ge_u16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_gt_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_gt_i16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_gt_u16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_le_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_le_i16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_le_u16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lg_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lt_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lt_i16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lt_u16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ne_i16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ne_u16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_neq_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nge_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ngt_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nle_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nlg_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nlt_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_o_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_t_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_tru_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_u_f16_e32 v255, v2 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_class_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_eq_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_eq_i16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_eq_u16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_f_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ge_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ge_i16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ge_u16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_gt_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_gt_i16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_gt_u16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_le_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_le_i16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_le_u16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lg_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lt_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lt_i16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lt_u16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ne_i16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ne_u16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_neq_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nge_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ngt_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nle_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nlg_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nlt_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_o_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_t_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_tru_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_u_f16_e32 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_class_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_eq_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_eq_i16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_eq_u16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_f_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ge_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ge_i16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ge_u16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_gt_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_gt_i16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_gt_u16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_le_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_le_i16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_le_u16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lg_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lt_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lt_i16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_lt_u16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ne_i16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ne_u16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_neq_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nge_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_ngt_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nle_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nlg_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_nlt_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_o_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_t_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_tru_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cmpx_u_f16_e32 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

