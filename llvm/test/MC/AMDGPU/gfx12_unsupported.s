// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// Unsupported instructions.
//===----------------------------------------------------------------------===//

s_waitcnt_expcnt exec_hi, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_waitcnt_lgkmcnt exec_hi, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_waitcnt_vmcnt exec_hi, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_waitcnt_vscnt exec_hi, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_subvector_loop_begin s0, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_subvector_loop_end s0, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_cdbgsys 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_cdbguser 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_cdbgsys_or_user 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_cdbgsys_and_user 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_legacy_f32 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2c_f32_f16 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v0, v1, v2 :: v_dual_max_f32 v3, v4, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v0, v1, v2 :: v_dual_min_f32 v3, v4, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_f32 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_rtn_f32 v0, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_f64 v0, v[1:2], v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_rtn_f64 v[0:1], v2, v[3:4], v[5:6]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_gs_reg_rtn v[0:1], v2 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_sub_gs_reg_rtn v[0:1], v2 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_wrap_rtn_b32 v0, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmpk_eq_i32 s0, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmpk_lg_i32 s0, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmpk_gt_i32 s0, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmpk_ge_i32 s0, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmpk_lt_i32 s0, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmpk_le_i32 s0, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmpk_eq_u32 s0, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmpk_lg_u32 s0, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmpk_gt_u32 s0, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmpk_ge_u32 s0, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmpk_lt_u32 s0, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cmpk_le_u32 s0, 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_inst_prefetch 1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_cmpswap_f32 v[5:6], off, s[96:99], s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_cmpswap_f32 v[5:6], off, s[96:99], s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_atomic_cmpswap_f32 v[5:6], off, s[96:99], s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_gws_sema_release_all gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_gws_init v0 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_gws_sema_v gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_gws_sema_br v0 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_gws_sema_p gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_gws_barrier v0 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_ordered_count v0, v1 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_f_f16 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_f16 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_f_f32 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_f32 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_f_f64 v[0:1], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_f64 v[0:1], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_f_i32 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_i32 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_f_u32 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_u32 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_f_i64 v[0:1], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_i64 v[0:1], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_f_u64 v[0:1], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmp_t_u64 v[0:1], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_f_f16 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_f16 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_f_f32 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_f32 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_f_f64 v[0:1], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_f64 v[0:1], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_f_i32 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_i32 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_f_u32 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_u32 v0, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_f_i64 v[0:1], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_i64 v[0:1], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_f_u64 v[0:1], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cmpx_t_u64 v[0:1], v[2:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_atomic_cmpswap_f32 v[5:6], off, s[96:99], s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_gl0_inv
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_gl1_inv
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_wbinvl1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_csub v1, v[0:1], v2 offset:64 th:TH_ATOMIC_RETURN
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid instruction

ds_add_f32 v255, v255 offset:4 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: gds modifier is not supported on this GPU

buffer_load_lds_b32 off, s[8:11], s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_lds_format_x off, s[8:11], s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_lds_i8 off, s[8:11], s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_lds_i16 off, s[8:11], s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_lds_u8 off, s[8:11], s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_lds_u16 off, s[8:11], s3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
