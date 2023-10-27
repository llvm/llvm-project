// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX11 %s

v_cmpx_class_f16 v1, v255
// GFX11: v_cmpx_class_f16_e64

v_cmpx_eq_f16 v1, v255
// GFX11: v_cmpx_eq_f16_e64

v_cmpx_eq_i16 v1, v255
// GFX11: v_cmpx_eq_i16_e64

v_cmpx_eq_u16 v1, v255
// GFX11: v_cmpx_eq_u16_e64

v_cmpx_f_f16 v1, v255
// GFX11: v_cmpx_f_f16_e64

v_cmpx_ge_f16 v1, v255
// GFX11: v_cmpx_ge_f16_e64

v_cmpx_ge_i16 v1, v255
// GFX11: v_cmpx_ge_i16_e64

v_cmpx_ge_u16 v1, v255
// GFX11: v_cmpx_ge_u16_e64

v_cmpx_gt_f16 v1, v255
// GFX11: v_cmpx_gt_f16_e64

v_cmpx_gt_i16 v1, v255
// GFX11: v_cmpx_gt_i16_e64

v_cmpx_gt_u16 v1, v255
// GFX11: v_cmpx_gt_u16_e64

v_cmpx_le_f16 v1, v255
// GFX11: v_cmpx_le_f16_e64

v_cmpx_le_i16 v1, v255
// GFX11: v_cmpx_le_i16_e64

v_cmpx_le_u16 v1, v255
// GFX11: v_cmpx_le_u16_e64

v_cmpx_lg_f16 v1, v255
// GFX11: v_cmpx_lg_f16_e64

v_cmpx_lt_f16 v1, v255
// GFX11: v_cmpx_lt_f16_e64

v_cmpx_lt_i16 v1, v255
// GFX11: v_cmpx_lt_i16_e64

v_cmpx_lt_u16 v1, v255
// GFX11: v_cmpx_lt_u16_e64

v_cmpx_ne_i16 v1, v255
// GFX11: v_cmpx_ne_i16_e64

v_cmpx_ne_u16 v1, v255
// GFX11: v_cmpx_ne_u16_e64

v_cmpx_neq_f16 v1, v255
// GFX11: v_cmpx_neq_f16_e64

v_cmpx_nge_f16 v1, v255
// GFX11: v_cmpx_nge_f16_e64

v_cmpx_ngt_f16 v1, v255
// GFX11: v_cmpx_ngt_f16_e64

v_cmpx_nle_f16 v1, v255
// GFX11: v_cmpx_nle_f16_e64

v_cmpx_nlg_f16 v1, v255
// GFX11: v_cmpx_nlg_f16_e64

v_cmpx_nlt_f16 v1, v255
// GFX11: v_cmpx_nlt_f16_e64

v_cmpx_o_f16 v1, v255
// GFX11: v_cmpx_o_f16_e64

v_cmpx_t_f16 v1, v255
// GFX11: v_cmpx_t_f16_e64

v_cmpx_tru_f16 v1, v255
// GFX11: v_cmpx_t_f16_e64

v_cmpx_u_f16 v1, v255
// GFX11: v_cmpx_u_f16_e64

v_cmpx_class_f16 v255, v2
// GFX11: v_cmpx_class_f16_e64

v_cmpx_eq_f16 v255, v2
// GFX11: v_cmpx_eq_f16_e64

v_cmpx_eq_i16 v255, v2
// GFX11: v_cmpx_eq_i16_e64

v_cmpx_eq_u16 v255, v2
// GFX11: v_cmpx_eq_u16_e64

v_cmpx_f_f16 v255, v2
// GFX11: v_cmpx_f_f16_e64

v_cmpx_ge_f16 v255, v2
// GFX11: v_cmpx_ge_f16_e64

v_cmpx_ge_i16 v255, v2
// GFX11: v_cmpx_ge_i16_e64

v_cmpx_ge_u16 v255, v2
// GFX11: v_cmpx_ge_u16_e64

v_cmpx_gt_f16 v255, v2
// GFX11: v_cmpx_gt_f16_e64

v_cmpx_gt_i16 v255, v2
// GFX11: v_cmpx_gt_i16_e64

v_cmpx_gt_u16 v255, v2
// GFX11: v_cmpx_gt_u16_e64

v_cmpx_le_f16 v255, v2
// GFX11: v_cmpx_le_f16_e64

v_cmpx_le_i16 v255, v2
// GFX11: v_cmpx_le_i16_e64

v_cmpx_le_u16 v255, v2
// GFX11: v_cmpx_le_u16_e64

v_cmpx_lg_f16 v255, v2
// GFX11: v_cmpx_lg_f16_e64

v_cmpx_lt_f16 v255, v2
// GFX11: v_cmpx_lt_f16_e64

v_cmpx_lt_i16 v255, v2
// GFX11: v_cmpx_lt_i16_e64

v_cmpx_lt_u16 v255, v2
// GFX11: v_cmpx_lt_u16_e64

v_cmpx_ne_i16 v255, v2
// GFX11: v_cmpx_ne_i16_e64

v_cmpx_ne_u16 v255, v2
// GFX11: v_cmpx_ne_u16_e64

v_cmpx_neq_f16 v255, v2
// GFX11: v_cmpx_neq_f16_e64

v_cmpx_nge_f16 v255, v2
// GFX11: v_cmpx_nge_f16_e64

v_cmpx_ngt_f16 v255, v2
// GFX11: v_cmpx_ngt_f16_e64

v_cmpx_nle_f16 v255, v2
// GFX11: v_cmpx_nle_f16_e64

v_cmpx_nlg_f16 v255, v2
// GFX11: v_cmpx_nlg_f16_e64

v_cmpx_nlt_f16 v255, v2
// GFX11: v_cmpx_nlt_f16_e64

v_cmpx_o_f16 v255, v2
// GFX11: v_cmpx_o_f16_e64

v_cmpx_t_f16 v255, v2
// GFX11: v_cmpx_t_f16_e64

v_cmpx_tru_f16 v255, v2
// GFX11: v_cmpx_t_f16_e64

v_cmpx_u_f16 v255, v2
// GFX11: v_cmpx_u_f16_e64

v_cmpx_class_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_class_f16_e64

v_cmpx_eq_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_eq_f16_e64

v_cmpx_eq_i16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_eq_i16_e64

v_cmpx_eq_u16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_eq_u16_e64

v_cmpx_f_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_f_f16_e64

v_cmpx_ge_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_ge_f16_e64

v_cmpx_ge_i16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_ge_i16_e64

v_cmpx_ge_u16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_ge_u16_e64

v_cmpx_gt_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_gt_f16_e64

v_cmpx_gt_i16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_gt_i16_e64

v_cmpx_gt_u16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_gt_u16_e64

v_cmpx_le_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_le_f16_e64

v_cmpx_le_i16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_le_i16_e64

v_cmpx_le_u16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_le_u16_e64

v_cmpx_lg_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_lg_f16_e64

v_cmpx_lt_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_lt_f16_e64

v_cmpx_lt_i16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_lt_i16_e64

v_cmpx_lt_u16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_lt_u16_e64

v_cmpx_ne_i16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_ne_i16_e64

v_cmpx_ne_u16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_ne_u16_e64

v_cmpx_neq_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_neq_f16_e64

v_cmpx_nge_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_nge_f16_e64

v_cmpx_ngt_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_ngt_f16_e64

v_cmpx_nle_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_nle_f16_e64

v_cmpx_nlg_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_nlg_f16_e64

v_cmpx_nlt_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_nlt_f16_e64

v_cmpx_o_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_o_f16_e64

v_cmpx_t_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_t_f16_e64

v_cmpx_tru_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_t_f16_e64

v_cmpx_u_f16 v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_u_f16_e64

v_cmpx_class_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_class_f16_e64

v_cmpx_eq_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_eq_f16_e64

v_cmpx_eq_i16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_eq_i16_e64

v_cmpx_eq_u16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_eq_u16_e64

v_cmpx_f_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_f_f16_e64

v_cmpx_ge_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_ge_f16_e64

v_cmpx_ge_i16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_ge_i16_e64

v_cmpx_ge_u16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_ge_u16_e64

v_cmpx_gt_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_gt_f16_e64

v_cmpx_gt_i16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_gt_i16_e64

v_cmpx_gt_u16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_gt_u16_e64

v_cmpx_le_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_le_f16_e64

v_cmpx_le_i16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_le_i16_e64

v_cmpx_le_u16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_le_u16_e64

v_cmpx_lg_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_lg_f16_e64

v_cmpx_lt_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_lt_f16_e64

v_cmpx_lt_i16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_lt_i16_e64

v_cmpx_lt_u16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_lt_u16_e64

v_cmpx_ne_i16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_ne_i16_e64

v_cmpx_ne_u16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_ne_u16_e64

v_cmpx_neq_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_neq_f16_e64

v_cmpx_nge_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_nge_f16_e64

v_cmpx_ngt_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_ngt_f16_e64

v_cmpx_nle_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_nle_f16_e64

v_cmpx_nlg_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_nlg_f16_e64

v_cmpx_nlt_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_nlt_f16_e64

v_cmpx_o_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_o_f16_e64

v_cmpx_t_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_t_f16_e64

v_cmpx_tru_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_t_f16_e64

v_cmpx_u_f16 v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_u_f16_e64

v_cmpx_class_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_class_f16_e64

v_cmpx_eq_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_eq_f16_e64

v_cmpx_eq_i16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_eq_i16_e64

v_cmpx_eq_u16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_eq_u16_e64

v_cmpx_f_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_f_f16_e64

v_cmpx_ge_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_ge_f16_e64

v_cmpx_ge_i16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_ge_i16_e64

v_cmpx_ge_u16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_ge_u16_e64

v_cmpx_gt_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_gt_f16_e64

v_cmpx_gt_i16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_gt_i16_e64

v_cmpx_gt_u16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_gt_u16_e64

v_cmpx_le_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_le_f16_e64

v_cmpx_le_i16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_le_i16_e64

v_cmpx_le_u16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_le_u16_e64

v_cmpx_lg_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_lg_f16_e64

v_cmpx_lt_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_lt_f16_e64

v_cmpx_lt_i16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_lt_i16_e64

v_cmpx_lt_u16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_lt_u16_e64

v_cmpx_ne_i16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_ne_i16_e64

v_cmpx_ne_u16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_ne_u16_e64

v_cmpx_neq_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_neq_f16_e64

v_cmpx_nge_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_nge_f16_e64

v_cmpx_ngt_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_ngt_f16_e64

v_cmpx_nle_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_nle_f16_e64

v_cmpx_nlg_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_nlg_f16_e64

v_cmpx_nlt_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_nlt_f16_e64

v_cmpx_o_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_o_f16_e64

v_cmpx_t_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_t_f16_e64

v_cmpx_tru_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_t_f16_e64

v_cmpx_u_f16 v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_u_f16_e64

v_cmpx_class_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_class_f16_e64

v_cmpx_eq_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_eq_f16_e64

v_cmpx_eq_i16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_eq_i16_e64

v_cmpx_eq_u16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_eq_u16_e64

v_cmpx_f_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_f_f16_e64

v_cmpx_ge_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_ge_f16_e64

v_cmpx_ge_i16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_ge_i16_e64

v_cmpx_ge_u16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_ge_u16_e64

v_cmpx_gt_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_gt_f16_e64

v_cmpx_gt_i16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_gt_i16_e64

v_cmpx_gt_u16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_gt_u16_e64

v_cmpx_le_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_le_f16_e64

v_cmpx_le_i16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_le_i16_e64

v_cmpx_le_u16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_le_u16_e64

v_cmpx_lg_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_lg_f16_e64

v_cmpx_lt_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_lt_f16_e64

v_cmpx_lt_i16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_lt_i16_e64

v_cmpx_lt_u16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_lt_u16_e64

v_cmpx_ne_i16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_ne_i16_e64

v_cmpx_ne_u16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_ne_u16_e64

v_cmpx_neq_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_neq_f16_e64

v_cmpx_nge_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_nge_f16_e64

v_cmpx_ngt_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_ngt_f16_e64

v_cmpx_nle_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_nle_f16_e64

v_cmpx_nlg_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_nlg_f16_e64

v_cmpx_nlt_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_nlt_f16_e64

v_cmpx_o_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_o_f16_e64

v_cmpx_t_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_t_f16_e64

v_cmpx_tru_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_t_f16_e64

v_cmpx_u_f16 v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_u_f16_e64

