// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// Unsupported instructions.
//===----------------------------------------------------------------------===//

s_cvt_f32_i32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_f32_u32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_u32_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_i32_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_f16_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_f32_f16 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_hi_f32_f16 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_trunc_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_ceil_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_rndne_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_floor_f32 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_floor_f16 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_ceil_f16 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_trunc_f16 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_rndne_f16 s5, s1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_add_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_sub_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_mul_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_min_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_max_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_fmac_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_fmamk_f32 s5, s1, 0x11213141, s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_fmaak_f32 s5, s1, s2, 0x11213141
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cvt_pk_rtz_f16_f32 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_add_f16 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_sub_f16 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_mul_f16 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_fmac_f16 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_max_f16 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_min_f16 s5, s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_lt_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_eq_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_le_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_gt_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_lg_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_ge_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_o_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_u_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nge_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nlg_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_ngt_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nle_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_neq_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nlt_f32 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_lt_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_eq_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_le_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_gt_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_lg_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_ge_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_o_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_u_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nge_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nlg_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_ngt_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nle_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_neq_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmp_nlt_f16 s1, s2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
