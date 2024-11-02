// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

v_add_co_ci_u32_sdwa v1, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_add_co_u32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_add_f16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_add_f32_sdwa v0, v0, v0 dst_unused:UNUSED_PAD src0_sel:WORD_1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_add_nc_u32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_and_b32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_ashrrev_i16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_ashrrev_i32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_bfrev_b32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_ceil_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_ceil_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_class_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_class_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_eq_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_eq_f32_sdwa exec, s2, v2 src0_sel:WORD_1 src1_sel:BYTE_2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_eq_i16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_eq_i32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_eq_u16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_eq_u32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_f_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_f_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_f_i32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_f_u32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_ge_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_ge_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_ge_i16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_ge_i32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_ge_u16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_ge_u32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_gt_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_gt_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_gt_i16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_gt_i32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_gt_u16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_gt_u32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_le_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_le_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_le_i16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_le_i32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_le_u16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_le_u32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_lg_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_lg_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_lt_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_lt_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_lt_i16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_lt_i32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_lt_u16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_lt_u32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_ne_i16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_ne_i32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_ne_u16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_ne_u32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_neq_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_neq_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_nge_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_nge_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_ngt_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_ngt_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_nle_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_nle_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_nlg_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_nlg_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_nlt_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_nlt_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_o_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_o_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_t_i32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_t_u32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_tru_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_tru_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_u_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmp_u_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_class_f16_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_class_f32_sdwa flat_scratch, v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_eq_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_eq_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_eq_i16_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_eq_i32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_eq_u16_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_eq_u32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_f_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_f_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_f_i32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_f_u32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_ge_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_ge_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_ge_i16_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_ge_i32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_ge_u16_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_ge_u32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_gt_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_gt_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_gt_i16_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_gt_i32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_gt_u16_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_gt_u32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_le_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_le_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_le_i16_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_le_i32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_le_u16_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_le_u32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_lg_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_lg_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_lt_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_lt_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_lt_i16_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_lt_i32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_lt_u16_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_lt_u32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_ne_i16_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_ne_i32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_ne_u16_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_ne_u32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_neq_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_neq_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_nge_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_nge_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_ngt_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_ngt_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_nle_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_nle_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_nlg_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_nlg_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_nlt_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_nlt_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_o_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_o_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_t_i32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_t_u32_sdwa exec_hi, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_tru_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_tru_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_u_f16_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cmpx_u_f32_sdwa -v1, v2 src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cndmask_b32_sdwa v255, v1, v2, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cos_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cos_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_f16_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_f16_i16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_f16_u16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_f32_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_f32_i32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_f32_u32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_f32_ubyte0_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_f32_ubyte1_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_f32_ubyte2_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_f32_ubyte3_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_flr_i32_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_i16_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_i32_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_norm_i16_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_norm_u16_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_off_f32_i4_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_pkrtz_f16_f32_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:BYTE_0 src1_sel:WORD_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_rpi_i32_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_u16_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_cvt_u32_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_dot2c_f32_f16_sdwa v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_exp_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_exp_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_ffbh_i32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_ffbh_u32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_ffbl_b32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_floor_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_floor_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_fmac_legacy_f32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_fract_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_fract_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_frexp_exp_i16_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_frexp_exp_i32_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_frexp_mant_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_frexp_mant_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_ldexp_f16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_log_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_log_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_lshlrev_b16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_lshlrev_b32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_lshrrev_b16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_lshrrev_b32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_max_f16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_max_f32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_max_i16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_max_i32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_max_u16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_max_u32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_min_f16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_min_f32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_min_i16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_min_i32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_min_u16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_min_u32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_mov_b32_sdwa v1, sext(-2+i1)
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_movreld_b32_sdwa v0, 64 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_movrels_b32_sdwa v0, 1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_movrelsd_2_b32_sdwa v0, 0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_movrelsd_b32_sdwa v0, 1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_mul_f16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_mul_f32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_mul_hi_i32_i24_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_mul_hi_u32_u24_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_mul_i32_i24_sdwa v1, v2, v3 clamp
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_mul_legacy_f32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_mul_lo_u16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_mul_u32_u24_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_nop_sdwa
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_not_b32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_or_b32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_rcp_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_rcp_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_rcp_iflag_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_rndne_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_rndne_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_rsq_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_rsq_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_sat_pk_u8_i16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_sin_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_sin_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_sqrt_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_sqrt_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_sub_co_ci_u32_sdwa v1, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_sub_co_u32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_sub_f16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_sub_f32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_sub_nc_u32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_subrev_co_ci_u32_sdwa v1, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_subrev_co_u32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_subrev_f16_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_subrev_f32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_subrev_nc_u32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_trunc_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_trunc_f32_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_xnor_b32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_xor_b32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
