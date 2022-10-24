// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=error %s

v_cmp_class_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_class_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_class_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_class_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_class_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_class_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_class_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_i16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_i16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_i16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_i16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_i16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_i16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_i16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_i16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_u16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_u16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_u16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_u16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_u16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_u16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_u16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_u16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_f_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_f_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_f_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_f_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_f_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_f_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_f_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_f_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_i16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_i16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_i16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_i16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_i16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_i16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_i16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_i16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_u16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_u16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_u16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_u16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_u16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_u16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_u16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_u16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_i16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_i16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_i16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_i16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_i16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_i16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_i16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_i16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_u16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_u16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_u16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_u16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_u16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_u16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_u16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_u16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_i16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_i16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_i16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_i16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_i16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_i16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_i16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_i16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_u16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_u16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_u16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_u16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_u16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_u16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_u16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_u16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lg_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lg_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lg_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lg_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lg_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lg_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lg_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lg_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_i16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_i16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_i16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_i16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_i16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_i16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_i16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_i16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_u16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_u16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_u16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_u16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_u16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_u16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_u16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_u16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_i16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_i16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_i16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_i16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_i16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_i16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_i16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_i16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_u16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_u16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_u16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_u16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_u16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_u16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_u16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_u16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_neq_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_neq_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_neq_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_neq_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_neq_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_neq_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_neq_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_neq_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nge_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nge_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nge_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nge_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nge_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nge_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nge_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nge_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nle_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nle_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nle_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nle_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nle_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nle_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nle_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nle_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_o_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_o_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_o_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_o_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_o_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_o_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_o_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_o_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_t_f16_e32 vcc, v1, v255
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc, v127, v255
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc, vcc_hi, v255
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc, vcc_lo, v255
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc_lo, v1, v255
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc_lo, v127, v255
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc, v1, v255
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc, v127, v255
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc, vcc_hi, v255
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc, vcc_lo, v255
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc_lo, v1, v255
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc_lo, v127, v255
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_u_f16_e32 vcc, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_u_f16_e32 vcc, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_u_f16_e32 vcc, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_u_f16_e32 vcc_lo, v1, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_u_f16_e32 vcc_lo, v127, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_u_f16_e32 vcc_lo, vcc_hi, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_u_f16_e32 vcc_lo, vcc_lo, v255
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_class_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_class_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_i16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_i16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_u16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_eq_u16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_f_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_f_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_i16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_i16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_u16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ge_u16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_i16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_i16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_u16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_gt_u16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_i16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_i16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_u16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_le_u16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lg_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lg_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_i16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_i16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_u16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_lt_u16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_i16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_i16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_u16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ne_u16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_neq_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_neq_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nge_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nge_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_ngt_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nle_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nle_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlg_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_nlt_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_o_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_o_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_t_f16_e32 vcc, v128, v2
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc_lo, v128, v2
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc, v128, v2
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc_lo, v128, v2
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_u_f16_e32 vcc_lo, v128, v2
// GFX11: error: operands are not valid for this GPU or mode

v_cmp_class_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_class_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_class_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_i16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_i16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_i16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_i16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_u16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_u16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_u16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_u16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_f_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_f_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_f_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_f_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_i16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_i16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_i16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_i16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_u16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_u16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_u16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_u16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_i16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_i16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_i16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_i16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_u16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_u16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_u16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_u16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_i16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_i16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_i16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_i16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_u16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_u16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_u16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_u16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lg_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lg_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lg_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lg_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_i16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_i16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_i16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_i16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_u16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_u16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_u16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_u16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_i16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_i16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_i16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_i16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_u16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_u16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_u16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_u16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_neq_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_neq_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_neq_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_neq_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nge_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nge_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nge_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nge_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ngt_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ngt_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ngt_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ngt_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nle_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nle_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nle_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nle_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlg_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlg_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlg_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlg_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlt_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlt_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlt_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlt_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_o_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_o_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_o_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_o_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc_lo, v1, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc_lo, v127, v255 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_class_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_class_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_i16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_i16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_u16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_u16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_f_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_f_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_i16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_i16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_u16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_u16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_i16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_i16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_u16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_u16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_i16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_i16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_u16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_u16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lg_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lg_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_i16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_i16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_u16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_u16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_i16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_i16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_u16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_u16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_neq_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_neq_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nge_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nge_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ngt_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ngt_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nle_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nle_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlg_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlg_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlt_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlt_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_o_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_o_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc_lo, v128, v2 quad_perm:[3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_class_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_class_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_class_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_i16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_i16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_i16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_i16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_u16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_u16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_u16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_u16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_f_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_f_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_f_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_f_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_i16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_i16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_i16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_i16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_u16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_u16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_u16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_u16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_i16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_i16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_i16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_i16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_u16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_u16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_u16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_u16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_i16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_i16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_i16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_i16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_u16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_u16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_u16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_u16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lg_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lg_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lg_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lg_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_i16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_i16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_i16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_i16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_u16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_u16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_u16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_u16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_i16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_i16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_i16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_i16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_u16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_u16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_u16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_u16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_neq_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_neq_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_neq_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_neq_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nge_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nge_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nge_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nge_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ngt_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ngt_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ngt_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ngt_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nle_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nle_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nle_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nle_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlg_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlg_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlg_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlg_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlt_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlt_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlt_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlt_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_o_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_o_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_o_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_o_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc_lo, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc_lo, v127, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_class_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_class_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_i16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_i16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_u16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_eq_u16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_f_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_f_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_i16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_i16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_u16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ge_u16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_i16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_i16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_u16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_gt_u16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_i16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_i16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_u16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_le_u16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lg_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lg_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_i16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_i16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_u16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_lt_u16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_i16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_i16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_u16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ne_u16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_neq_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_neq_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nge_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nge_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ngt_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_ngt_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nle_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nle_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlg_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlg_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlt_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_nlt_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_o_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_o_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_t_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_tru_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

v_cmp_u_f16_e32 vcc_lo, v128, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: error: invalid operand for instruction

