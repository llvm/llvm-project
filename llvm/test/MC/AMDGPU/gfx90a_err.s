// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck --check-prefix=GFX90A --implicit-check-not=error: %s

ds_add_src2_u32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_src2_f32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_sub_src2_u32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_rsub_src2_u32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_inc_src2_u32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_dec_src2_u32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_i32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_i32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_u32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_u32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_and_src2_b32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_or_src2_b32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_xor_src2_b32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_f32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_f32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_src2_u64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_sub_src2_u64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_rsub_src2_u64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_inc_src2_u64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_dec_src2_u64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_i64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_i64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_u64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_u64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_and_src2_b64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_or_src2_b64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_xor_src2_b64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_min_src2_f64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_max_src2_f64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_write_src2_b32 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_write_src2_b64 v1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4 v[5:8], v1, s[8:15], s[12:15]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_gather4h v[251:254], v[1:2], s[8:15], s[12:15] dmask:0x1
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_get_lod v5, v1, s[8:15], s[12:15]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_legacy_f32_e32 v5, v1, v2
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: e32 variant of this instruction is not supported

v_mul_legacy_f32_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_mul_legacy_f32_dpp v5, v1, v2  quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_interp_p1_f32 v5, v1, attr0.x
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p1_f32_e64 v5, v2, attr0.x
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p2_f32 v5, v1, attr0.x
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_mov_f32 v5, p10, attr0.x
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p1ll_f16 v5, v2, attr0.x
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p1lv_f16 v5, v2, attr0.x, v3
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p2_legacy_f16 v5, v2, attr0.x, v3
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_interp_p2_f16 v5, v2, attr0.x, v3
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mov_b32_dpp v5, v1 row_share:1 row_mask:0x0 bank_mask:0x0
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

v_ceil_f64_dpp v[0:1], v[2:3] quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_newbcast

v_ceil_f64_dpp v[0:1], v[2:3] row_shl:1 row_mask:0xf bank_mask:0xf
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_newbcast

v_ceil_f64_dpp v[0:1], v[2:3] wave_ror:1 row_mask:0xf bank_mask:0xf
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_newbcast

v_cvt_u32_f64 v5, v[0:1] quad_perm:[0,2,1,1] row_mask:0xf bank_mask:0xf
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_newbcast

v_ceil_f64_dpp v[0:1], v[2:3] row_share:1 row_mask:0xf bank_mask:0xf
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_add v2, v[2:3], a2 glc
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

flat_atomic_add a2, v[2:3], v2 glc
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_xyzw v[0:3], off, s[4:7],  dfmt:15,  nfmt:2, s1 tfe
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_dwordx4 v[0:3], off, s[12:15], s4 offset:4095 glc tfe
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

ds_write2_b64 v1, a[4:5], v[2:3] offset1:255
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

ds_write2_b64 v1, v[4:5], a[2:3] offset1:255
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

ds_wrxchg2st64_rtn_b32 v[6:7], v1, a2, a3 offset0:127
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_load v[0:4], v2, s[0:7] dmask:0xf unorm tfe
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_sample_lz v[0:3], v[0:1], s[4:11], s[16:19] dmask:0xf
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_d v[0:3], v[0:1], s[4:11], s[16:19] dmask:0xf
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_o v[0:3], v[0:1], s[4:11], s[16:19] dmask:0xf
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cl v[0:3], v[0:1], s[4:11], s[16:19] dmask:0xf
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_cd v[0:3], v[0:1], s[4:11], s[16:19] dmask:0xf
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_sample_b v[0:3], v[0:1], s[4:11], s[16:19] dmask:0xf
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mov_b32_sdwa v1, src_lds_direct dst_sel:DWORD
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct is not supported on this GPU

v_add_f32_sdwa v5, v1, lds_direct dst_sel:DWORD
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct is not supported on this GPU

v_ashrrev_i16 v0, lds_direct, v0
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct is not supported on this GPU

v_add_f32 v5, v1, lds_direct
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: lds_direct is not supported on this GPU

ds_gws_init a1 offset:65535 gds
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: vgpr must be even aligned

ds_gws_init a255 offset:65535 gds
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: vgpr must be even aligned

ds_gws_sema_br v1 offset:65535 gds
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: vgpr must be even aligned

ds_gws_sema_br v255 offset:65535 gds
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: vgpr must be even aligned

ds_gws_barrier a3 offset:4 gds
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: vgpr must be even aligned

ds_gws_barrier a255 offset:4 gds
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: vgpr must be even aligned

ds_ordered_count v5, v1 offset:65535 gds
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

exp pos0 v3, v2, v1, v0
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_lds_dword v[2:3], off
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_lds_dword v2, off
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_read_b32 v0, v1 gds
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: gds modifier is not supported on this GPU

// op_sel not allowed in dot opcodes with 4- or 8-bit packed data

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot4c_i32_i8 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8_u32_u4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[0,0] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[0,1] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,0]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_dot8c_i32_i4 v0, v1, v2, v3 op_sel:[1,1] op_sel_hi:[1,1]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

