// RUN: llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s | FileCheck -check-prefix=VI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck -check-prefix=NOSICI %s
    // RUN: not llvm-mc -arch=amdgcn -mcpu=hawaii -show-encoding %s 2>&1 | FileCheck -check-prefix=NOSICI %s

v_cmp_class_f16 vcc, v2, v4
// VI: v_cmp_class_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x28,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_class_f16 vcc, v2, v4
// VI: v_cmpx_class_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x2a,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_f_f16 vcc, v2, v4
// VI: v_cmp_f_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x40,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_lt_f16 vcc, v2, v4
// VI: v_cmp_lt_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x42,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_eq_f16 vcc, v2, v4
// VI: v_cmp_eq_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x44,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_le_f16 vcc, v2, v4
// VI: v_cmp_le_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x46,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_gt_f16 vcc, v2, v4
// VI: v_cmp_gt_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x48,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_lg_f16 vcc, v2, v4
// VI: v_cmp_lg_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x4a,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_ge_f16 vcc, v2, v4
// VI: v_cmp_ge_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x4c,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_o_f16 vcc, v2, v4
// VI: v_cmp_o_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x4e,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_u_f16 vcc, v2, v4
// VI: v_cmp_u_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x50,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_nge_f16 vcc, v2, v4
// VI: v_cmp_nge_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x52,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_nlg_f16 vcc, v2, v4
// VI: v_cmp_nlg_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x54,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_ngt_f16 vcc, v2, v4
// VI: v_cmp_ngt_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x56,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_nle_f16 vcc, v2, v4
// VI: v_cmp_nle_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x58,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_neq_f16 vcc, v2, v4
// VI: v_cmp_neq_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x5a,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_nlt_f16 vcc, v2, v4
// VI: v_cmp_nlt_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x5c,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_tru_f16 vcc, v2, v4
// VI: v_cmp_tru_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x5e,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_f_f16 vcc, v2, v4
// VI: v_cmpx_f_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x60,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_lt_f16 vcc, v2, v4
// VI: v_cmpx_lt_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x62,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_eq_f16 vcc, v2, v4
// VI: v_cmpx_eq_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x64,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_le_f16 vcc, v2, v4
// VI: v_cmpx_le_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x66,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_gt_f16 vcc, v2, v4
// VI: v_cmpx_gt_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x68,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_lg_f16 vcc, v2, v4
// VI: v_cmpx_lg_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x6a,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_ge_f16 vcc, v2, v4
// VI: v_cmpx_ge_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x6c,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_o_f16 vcc, v2, v4
// VI: v_cmpx_o_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x6e,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_u_f16 vcc, v2, v4
// VI: v_cmpx_u_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x70,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_nge_f16 vcc, v2, v4
// VI: v_cmpx_nge_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x72,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_nlg_f16 vcc, v2, v4
// VI: v_cmpx_nlg_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x74,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_ngt_f16 vcc, v2, v4
// VI: v_cmpx_ngt_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x76,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_nle_f16 vcc, v2, v4
// VI: v_cmpx_nle_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x78,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_neq_f16 vcc, v2, v4
// VI: v_cmpx_neq_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x7a,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_nlt_f16 vcc, v2, v4
// VI: v_cmpx_nlt_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x7c,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_tru_f16 vcc, v2, v4
// VI: v_cmpx_tru_f16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x7e,0x7c]
// NOSICI: error: instruction not supported on this GPU

v_cmp_f_i16 vcc, v2, v4
// VI: v_cmp_f_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x40,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_lt_i16 vcc, v2, v4
// VI: v_cmp_lt_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x42,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_eq_i16 vcc, v2, v4
// VI: v_cmp_eq_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x44,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_le_i16 vcc, v2, v4
// VI: v_cmp_le_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x46,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_gt_i16 vcc, v2, v4
// VI: v_cmp_gt_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x48,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_ne_i16 vcc, v2, v4
// VI: v_cmp_ne_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x4a,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_ge_i16 vcc, v2, v4
// VI: v_cmp_ge_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x4c,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_t_i16 vcc, v2, v4
// VI: v_cmp_t_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x4e,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_f_u16 vcc, v2, v4
// VI: v_cmp_f_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x50,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_lt_u16 vcc, v2, v4
// VI: v_cmp_lt_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x52,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_eq_u16 vcc, v2, v4
// VI: v_cmp_eq_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x54,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_le_u16 vcc, v2, v4
// VI: v_cmp_le_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x56,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_gt_u16 vcc, v2, v4
// VI: v_cmp_gt_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x58,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_ne_u16 vcc, v2, v4
// VI: v_cmp_ne_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x5a,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_ge_u16 vcc, v2, v4
// VI: v_cmp_ge_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x5c,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmp_t_u16 vcc, v2, v4
// VI: v_cmp_t_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x5e,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_f_i16 vcc, v2, v4
// VI: v_cmpx_f_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x60,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_lt_i16 vcc, v2, v4
// VI: v_cmpx_lt_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x62,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_eq_i16 vcc, v2, v4
// VI: v_cmpx_eq_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x64,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_le_i16 vcc, v2, v4
// VI: v_cmpx_le_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x66,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_gt_i16 vcc, v2, v4
// VI: v_cmpx_gt_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x68,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_ne_i16 vcc, v2, v4
// VI: v_cmpx_ne_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x6a,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_ge_i16 vcc, v2, v4
// VI: v_cmpx_ge_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x6c,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_t_i16 vcc, v2, v4
// VI: v_cmpx_t_i16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x6e,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_f_u16 vcc, v2, v4
// VI: v_cmpx_f_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x70,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_lt_u16 vcc, v2, v4
// VI: v_cmpx_lt_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x72,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_eq_u16 vcc, v2, v4
// VI: v_cmpx_eq_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x74,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_le_u16 vcc, v2, v4
// VI: v_cmpx_le_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x76,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_gt_u16 vcc, v2, v4
// VI: v_cmpx_gt_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x78,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_ne_u16 vcc, v2, v4
// VI: v_cmpx_ne_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x7a,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_ge_u16 vcc, v2, v4
// VI: v_cmpx_ge_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x7c,0x7d]
// NOSICI: error: instruction not supported on this GPU

v_cmpx_t_u16 vcc, v2, v4
// VI: v_cmpx_t_u16_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x7e,0x7d]
// NOSICI: error: instruction not supported on this GPU
