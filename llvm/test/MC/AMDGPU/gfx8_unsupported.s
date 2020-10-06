// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// Unsupported instructions.
//===----------------------------------------------------------------------===//

buffer_atomic_add_f32 v255, off, s[8:11], s3 offset:4095
// CHECK: error: instruction not supported on this GPU

buffer_atomic_fcmpswap v[0:1], off, s[0:3], s0 offset:4095
// CHECK: error: instruction not supported on this GPU

buffer_atomic_fcmpswap_x2 v[0:3], off, s[0:3], s0 offset:4095
// CHECK: error: instruction not supported on this GPU

buffer_atomic_fmax v0, off, s[0:3], s0 offset:4095 glc
// CHECK: error: instruction not supported on this GPU

buffer_atomic_fmax_x2 v[0:1], v0, s[0:3], s0 idxen offset:4095
// CHECK: error: instruction not supported on this GPU

buffer_atomic_fmin v0, off, s[0:3], s0
// CHECK: error: instruction not supported on this GPU

buffer_atomic_fmin_x2 v[0:1], off, s[0:3], s0 offset:4095 slc
// CHECK: error: instruction not supported on this GPU

buffer_atomic_pk_add_f16 v255, off, s[8:11], s3 offset:4095
// CHECK: error: instruction not supported on this GPU

buffer_gl0_inv
// CHECK: error: instruction not supported on this GPU

buffer_gl1_inv
// CHECK: error: instruction not supported on this GPU

buffer_load_format_d16_hi_x v5, off, s[8:11], s3
// CHECK: error: instruction not supported on this GPU

buffer_load_sbyte_d16 v1, off, s[4:7], s1
// CHECK: error: instruction not supported on this GPU

buffer_load_sbyte_d16_hi v1, off, s[4:7], s1
// CHECK: error: instruction not supported on this GPU

buffer_load_short_d16 v1, off, s[4:7], s1
// CHECK: error: instruction not supported on this GPU

buffer_load_short_d16_hi v1, off, s[4:7], s1
// CHECK: error: instruction not supported on this GPU

buffer_load_ubyte_d16 v1, off, s[4:7], s1
// CHECK: error: instruction not supported on this GPU

buffer_load_ubyte_d16_hi v1, off, s[4:7], s1
// CHECK: error: instruction not supported on this GPU

buffer_store_byte_d16_hi v1, off, s[12:15], -1 offset:4095
// CHECK: error: instruction not supported on this GPU

buffer_store_format_d16_hi_x v1, off, s[12:15], s4 offset:4095 glc
// CHECK: error: instruction not supported on this GPU

buffer_store_short_d16_hi v1, off, s[12:15], -1 offset:4095
// CHECK: error: instruction not supported on this GPU

ds_read_addtid_b32 v255 offset:65535
// CHECK: error: instruction not supported on this GPU

ds_read_i8_d16 v255, v1 offset:65535
// CHECK: error: instruction not supported on this GPU

ds_read_i8_d16_hi v255, v1 offset:65535
// CHECK: error: instruction not supported on this GPU

ds_read_u16_d16 v255, v1 offset:65535
// CHECK: error: instruction not supported on this GPU

ds_read_u16_d16_hi v255, v1 offset:65535
// CHECK: error: instruction not supported on this GPU

ds_read_u8_d16 v255, v1 offset:65535
// CHECK: error: instruction not supported on this GPU

ds_read_u8_d16_hi v255, v1 offset:65535
// CHECK: error: instruction not supported on this GPU

ds_write_addtid_b32 v255 offset:65535
// CHECK: error: instruction not supported on this GPU

ds_write_b16_d16_hi v1, v2
// CHECK: error: instruction not supported on this GPU

ds_write_b8_d16_hi v1, v2
// CHECK: error: instruction not supported on this GPU

flat_atomic_fcmpswap v0, v[1:2], v[2:3] glc
// CHECK: error: instruction not supported on this GPU

flat_atomic_fcmpswap_x2 v[0:1], v[1:2], v[2:5] glc
// CHECK: error: instruction not supported on this GPU

flat_atomic_fmax v0, v[1:2], v2 glc
// CHECK: error: instruction not supported on this GPU

flat_atomic_fmax_x2 v[0:1], v[1:2], v[2:3] glc
// CHECK: error: instruction not supported on this GPU

flat_atomic_fmin v0, v[1:2], v2 glc
// CHECK: error: instruction not supported on this GPU

flat_atomic_fmin_x2 v[0:1], v[1:2], v[2:3] glc
// CHECK: error: instruction not supported on this GPU

flat_load_sbyte_d16 v1, v[3:4]
// CHECK: error: instruction not supported on this GPU

flat_load_sbyte_d16_hi v1, v[3:4]
// CHECK: error: instruction not supported on this GPU

flat_load_short_d16 v1, v[3:4]
// CHECK: error: instruction not supported on this GPU

flat_load_short_d16_hi v1, v[3:4]
// CHECK: error: instruction not supported on this GPU

flat_load_ubyte_d16 v1, v[3:4]
// CHECK: error: instruction not supported on this GPU

flat_load_ubyte_d16_hi v1, v[3:4]
// CHECK: error: instruction not supported on this GPU

flat_store_byte_d16_hi v[1:2], v2
// CHECK: error: instruction not supported on this GPU

flat_store_short_d16_hi v[1:2], v2
// CHECK: error: instruction not supported on this GPU

global_atomic_add v0, v[1:2], v2, off glc slc
// CHECK: error: instruction not supported on this GPU

global_atomic_add_f32 v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_atomic_add_x2 v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_atomic_and v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_atomic_and_x2 v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_atomic_cmpswap v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_atomic_cmpswap_x2 v[1:2], v[252:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_atomic_dec v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_atomic_dec_x2 v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_atomic_inc v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_atomic_inc_x2 v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_atomic_or v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_atomic_or_x2 v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_atomic_pk_add_f16 v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_atomic_smax v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_atomic_smax_x2 v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_atomic_smin v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_atomic_smin_x2 v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_atomic_sub v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_atomic_sub_x2 v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_atomic_swap v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_atomic_swap_x2 v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_atomic_umax v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_atomic_umax_x2 v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_atomic_umin v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_atomic_umin_x2 v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_atomic_xor v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_atomic_xor_x2 v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_load_dword v1, v3, s[2:3]
// CHECK: error: instruction not supported on this GPU

global_load_dwordx2 v[1:2], v[3:4], off
// CHECK: error: instruction not supported on this GPU

global_load_dwordx3 v[1:3], v[3:4], off
// CHECK: error: instruction not supported on this GPU

global_load_dwordx4 v[1:4], v[3:4], off
// CHECK: error: instruction not supported on this GPU

global_load_sbyte v1, v[3:4], off
// CHECK: error: instruction not supported on this GPU

global_load_sbyte_d16 v1, v[3:4], off
// CHECK: error: instruction not supported on this GPU

global_load_sbyte_d16_hi v1, v[3:4], off
// CHECK: error: instruction not supported on this GPU

global_load_short_d16 v1, v[3:4], off
// CHECK: error: instruction not supported on this GPU

global_load_short_d16_hi v1, v[3:4], off
// CHECK: error: instruction not supported on this GPU

global_load_sshort v1, v[3:4], off
// CHECK: error: instruction not supported on this GPU

global_load_ubyte v1, v[3:4], off
// CHECK: error: instruction not supported on this GPU

global_load_ubyte_d16 v1, v[3:4], off
// CHECK: error: instruction not supported on this GPU

global_load_ubyte_d16_hi v1, v[3:4], off
// CHECK: error: instruction not supported on this GPU

global_load_ushort v1, v[3:4], off
// CHECK: error: instruction not supported on this GPU

global_store_byte v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_store_byte_d16_hi v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_store_dword v254, v1, s[2:3] offset:16
// CHECK: error: instruction not supported on this GPU

global_store_dwordx2 v[1:2], v[254:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_store_dwordx3 v[1:2], v[253:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_store_dwordx4 v[1:2], v[252:255], off offset:-1
// CHECK: error: instruction not supported on this GPU

global_store_short v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

global_store_short_d16_hi v[1:2], v2, off
// CHECK: error: instruction not supported on this GPU

s_and_saveexec_b32 exec_hi, s1
// CHECK: error: instruction not supported on this GPU

s_andn1_saveexec_b32 exec_hi, s1
// CHECK: error: instruction not supported on this GPU

s_andn1_saveexec_b64 exec, s[2:3]
// CHECK: error: instruction not supported on this GPU

s_andn1_wrexec_b32 exec_hi, s1
// CHECK: error: instruction not supported on this GPU

s_andn1_wrexec_b64 exec, s[2:3]
// CHECK: error: instruction not supported on this GPU

s_andn2_saveexec_b32 exec_hi, s1
// CHECK: error: instruction not supported on this GPU

s_andn2_wrexec_b32 exec_hi, s1
// CHECK: error: instruction not supported on this GPU

s_andn2_wrexec_b64 exec, s[2:3]
// CHECK: error: instruction not supported on this GPU

s_atomic_add s5, s[2:3], 0x0
// CHECK: error: instruction not supported on this GPU

s_atomic_add_x2 s[10:11], s[2:3], s101
// CHECK: error: instruction not supported on this GPU

s_atomic_and s5, s[2:3], s101
// CHECK: error: instruction not supported on this GPU

s_atomic_and_x2 s[10:11], s[2:3], 0x0
// CHECK: error: instruction not supported on this GPU

s_atomic_cmpswap s[10:11], s[2:3], 0x0
// CHECK: error: instruction not supported on this GPU

s_atomic_cmpswap_x2 s[20:23], s[2:3], 0x0
// CHECK: error: instruction not supported on this GPU

s_atomic_dec s5, s[2:3], s0 glc
// CHECK: error: instruction not supported on this GPU

s_atomic_dec_x2 s[10:11], s[2:3], s101
// CHECK: error: instruction not supported on this GPU

s_atomic_inc s5, s[2:3], s0 glc
// CHECK: error: instruction not supported on this GPU

s_atomic_inc_x2 s[10:11], s[2:3], s101
// CHECK: error: instruction not supported on this GPU

s_atomic_or s5, s[2:3], 0x0
// CHECK: error: instruction not supported on this GPU

s_atomic_or_x2 s[10:11], s[2:3], s0 glc
// CHECK: error: instruction not supported on this GPU

s_atomic_smax s5, s[2:3], s101
// CHECK: error: instruction not supported on this GPU

s_atomic_smax_x2 s[10:11], s[2:3], s0 glc
// CHECK: error: instruction not supported on this GPU

s_atomic_smin s5, s[2:3], s101
// CHECK: error: instruction not supported on this GPU

s_atomic_smin_x2 s[10:11], s[2:3], s0 glc
// CHECK: error: instruction not supported on this GPU

s_atomic_sub s5, s[2:3], s101
// CHECK: error: instruction not supported on this GPU

s_atomic_sub_x2 s[10:11], s[2:3], s0 glc
// CHECK: error: instruction not supported on this GPU

s_atomic_swap s5, s[2:3], -1
// CHECK: error: instruction not supported on this GPU

s_atomic_swap_x2 s[10:11], s[2:3], s0 glc
// CHECK: error: instruction not supported on this GPU

s_atomic_umax s5, s[2:3], s0 glc
// CHECK: error: instruction not supported on this GPU

s_atomic_umax_x2 s[10:11], s[2:3], s101
// CHECK: error: instruction not supported on this GPU

s_atomic_umin s5, s[2:3], s101
// CHECK: error: instruction not supported on this GPU

s_atomic_umin_x2 s[10:11], s[2:3], s0 glc
// CHECK: error: instruction not supported on this GPU

s_atomic_xor s5, s[2:3], s101
// CHECK: error: instruction not supported on this GPU

s_atomic_xor_x2 s[10:11], s[2:3], s0 glc
// CHECK: error: instruction not supported on this GPU

s_bitreplicate_b64_b32 exec, s2
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_add s5, s[4:7], 0x0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_add_x2 s[10:11], s[4:7], s0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_and s101, s[4:7], s0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_and_x2 s[10:11], s[8:11], s0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap s[10:11], s[4:7], 0x0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], 0x0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_dec s5, s[4:7], s0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_dec_x2 s[10:11], s[4:7], s0 glc
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_inc s101, s[4:7], s0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_inc_x2 s[10:11], s[4:7], 0x0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_or s5, s[8:11], s0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_or_x2 s[10:11], s[96:99], s0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_smax s5, s[4:7], s101
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_smax_x2 s[100:101], s[4:7], s0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_smin s5, s[4:7], 0x0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_smin_x2 s[12:13], s[4:7], s0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_sub s5, s[4:7], s0 glc
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_sub_x2 s[10:11], s[4:7], s0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_swap s5, s[4:7], -1
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_swap_x2 s[10:11], s[4:7], s0 glc
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_umax s5, s[4:7], s0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_umax_x2 s[10:11], s[4:7], s0 glc
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_umin s5, s[4:7], s0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_umin_x2 s[10:11], s[4:7], s0 glc
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_xor s5, s[4:7], s0
// CHECK: error: instruction not supported on this GPU

s_buffer_atomic_xor_x2 s[10:11], s[4:7], s0 glc
// CHECK: error: instruction not supported on this GPU

s_call_b64 exec, 0x1234
// CHECK: error: instruction not supported on this GPU

s_clause 0x0
// CHECK: error: instruction not supported on this GPU

s_code_end
// CHECK: error: instruction not supported on this GPU

s_dcache_discard s[2:3], 0x0
// CHECK: error: instruction not supported on this GPU

s_dcache_discard_x2 s[2:3], 0x0
// CHECK: error: instruction not supported on this GPU

s_denorm_mode 0x0
// CHECK: error: instruction not supported on this GPU

s_endpgm_ordered_ps_done
// CHECK: error: instruction not supported on this GPU

s_get_waveid_in_workgroup s0
// CHECK: error: instruction not supported on this GPU

s_gl1_inv
// CHECK: error: instruction not supported on this GPU

s_inst_prefetch 0x0
// CHECK: error: instruction not supported on this GPU

s_lshl1_add_u32 exec_hi, s1, s2
// CHECK: error: instruction not supported on this GPU

s_lshl2_add_u32 exec_hi, s1, s2
// CHECK: error: instruction not supported on this GPU

s_lshl3_add_u32 exec_hi, s1, s2
// CHECK: error: instruction not supported on this GPU

s_lshl4_add_u32 exec_hi, s1, s2
// CHECK: error: instruction not supported on this GPU

s_movrelsd_2_b32 s0, s1
// CHECK: error: instruction not supported on this GPU

s_mul_hi_i32 exec_hi, s1, s2
// CHECK: error: instruction not supported on this GPU

s_mul_hi_u32 exec_hi, s1, s2
// CHECK: error: instruction not supported on this GPU

s_nand_saveexec_b32 exec_hi, s1
// CHECK: error: instruction not supported on this GPU

s_nor_saveexec_b32 exec_hi, s1
// CHECK: error: instruction not supported on this GPU

s_or_saveexec_b32 exec_hi, s1
// CHECK: error: instruction not supported on this GPU

s_orn1_saveexec_b32 exec_hi, s1
// CHECK: error: instruction not supported on this GPU

s_orn1_saveexec_b64 exec, s[2:3]
// CHECK: error: instruction not supported on this GPU

s_orn2_saveexec_b32 exec_hi, s1
// CHECK: error: instruction not supported on this GPU

s_pack_hh_b32_b16 exec_hi, s1, s2
// CHECK: error: instruction not supported on this GPU

s_pack_lh_b32_b16 exec_hi, s1, s2
// CHECK: error: instruction not supported on this GPU

s_pack_ll_b32_b16 exec_hi, s1, s2
// CHECK: error: instruction not supported on this GPU

s_round_mode 0x0
// CHECK: error: instruction not supported on this GPU

s_scratch_load_dword s5, s[2:3], s0 glc
// CHECK: error: instruction not supported on this GPU

s_scratch_load_dwordx2 s[100:101], s[2:3], s0
// CHECK: error: instruction not supported on this GPU

s_scratch_load_dwordx4 s[20:23], s[4:5], s0
// CHECK: error: instruction not supported on this GPU

s_scratch_store_dword s1, s[4:5], 0x123 glc
// CHECK: error: instruction not supported on this GPU

s_scratch_store_dwordx2 s[2:3], s[4:5], s101 glc
// CHECK: error: instruction not supported on this GPU

s_scratch_store_dwordx4 s[4:7], s[4:5], s0 glc
// CHECK: error: instruction not supported on this GPU

s_subvector_loop_begin exec_hi, 0x1234
// CHECK: error: instruction not supported on this GPU

s_subvector_loop_end exec_hi, 0x1234
// CHECK: error: instruction not supported on this GPU

s_ttracedata_imm 0x0
// CHECK: error: instruction not supported on this GPU

s_version 0x1234
// CHECK: error: instruction not supported on this GPU

s_waitcnt_expcnt exec_hi, 0x1234
// CHECK: error: instruction not supported on this GPU

s_waitcnt_lgkmcnt exec_hi, 0x1234
// CHECK: error: instruction not supported on this GPU

s_waitcnt_vmcnt exec_hi, 0x1234
// CHECK: error: instruction not supported on this GPU

s_waitcnt_vscnt exec_hi, 0x1234
// CHECK: error: instruction not supported on this GPU

s_xnor_saveexec_b32 exec_hi, s1
// CHECK: error: instruction not supported on this GPU

s_xor_saveexec_b32 exec_hi, s1
// CHECK: error: instruction not supported on this GPU

scratch_load_dword v0, v1, off offset:-2048 glc slc
// CHECK: error: instruction not supported on this GPU

scratch_load_dwordx2 v[1:2], v3, off
// CHECK: error: instruction not supported on this GPU

scratch_load_dwordx3 v[1:3], v4, off
// CHECK: error: instruction not supported on this GPU

scratch_load_dwordx4 v[1:4], v5, off
// CHECK: error: instruction not supported on this GPU

scratch_load_sbyte v1, v2, off
// CHECK: error: instruction not supported on this GPU

scratch_load_sbyte_d16 v1, v2, off
// CHECK: error: instruction not supported on this GPU

scratch_load_sbyte_d16_hi v1, v2, off
// CHECK: error: instruction not supported on this GPU

scratch_load_short_d16 v1, v2, off
// CHECK: error: instruction not supported on this GPU

scratch_load_short_d16_hi v1, v2, off
// CHECK: error: instruction not supported on this GPU

scratch_load_sshort v1, v2, off
// CHECK: error: instruction not supported on this GPU

scratch_load_ubyte v1, v2, off
// CHECK: error: instruction not supported on this GPU

scratch_load_ubyte_d16 v1, v2, off
// CHECK: error: instruction not supported on this GPU

scratch_load_ubyte_d16_hi v1, v2, off
// CHECK: error: instruction not supported on this GPU

scratch_load_ushort v1, v2, off
// CHECK: error: instruction not supported on this GPU

scratch_store_byte off, v2, flat_scratch_hi offset:-1
// CHECK: error: instruction not supported on this GPU

scratch_store_byte_d16_hi off, v2, flat_scratch_hi offset:-1
// CHECK: error: instruction not supported on this GPU

scratch_store_dword off, v2, exec_hi
// CHECK: error: instruction not supported on this GPU

scratch_store_dwordx2 off, v[254:255], s3 offset:-1
// CHECK: error: instruction not supported on this GPU

scratch_store_dwordx3 off, v[253:255], s3 offset:-1
// CHECK: error: instruction not supported on this GPU

scratch_store_dwordx4 off, v[252:255], s3 offset:-1
// CHECK: error: instruction not supported on this GPU

scratch_store_short off, v2, flat_scratch_hi offset:-1
// CHECK: error: instruction not supported on this GPU

scratch_store_short_d16_hi off, v2, flat_scratch_hi offset:-1
// CHECK: error: instruction not supported on this GPU

v_accvgpr_read_b32 a0, a0
// CHECK: error: instruction not supported on this GPU

v_accvgpr_write_b32 a0, 65
// CHECK: error: instruction not supported on this GPU

v_add3_u32 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_add_co_ci_u32 v1, sext(v1), sext(v4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// CHECK: error: instruction not supported on this GPU

v_add_co_ci_u32_dpp v0, vcc, v0, v0, vcc dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: error: instruction not supported on this GPU

v_add_co_ci_u32_e32 v255, vcc, v1, v2, vcc
// CHECK: error: instruction not supported on this GPU

v_add_co_ci_u32_e64 v255, s12, v1, v2, s6
// CHECK: error: instruction not supported on this GPU

v_add_co_ci_u32_sdwa v1, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// CHECK: error: instruction not supported on this GPU

v_add_co_u32 v0, exec, v0, v2
// CHECK: error: instruction not supported on this GPU

v_add_co_u32_dpp v255, vcc, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_add_co_u32_e32 v2, vcc, s0, v2
// CHECK: error: instruction not supported on this GPU

v_add_co_u32_e64 v0, s0, v0, v2
// CHECK: error: instruction not supported on this GPU

v_add_co_u32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// CHECK: error: instruction not supported on this GPU

v_add_i16 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_add_i32 lds_direct, v0, v0
// CHECK: error: instruction not supported on this GPU

v_add_i32_e32 v0, vcc, 0.5, v0
// CHECK: error: instruction not supported on this GPU

v_add_i32_e64 v1, s[0:1], v2, v3
// CHECK: error: instruction not supported on this GPU

v_add_lshl_u32 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_add_nc_i16 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_add_nc_i32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_add_nc_u16 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_add_nc_u32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: error: instruction not supported on this GPU

v_add_nc_u32_e32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_add_nc_u32_e64 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_add_nc_u32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: error: instruction not supported on this GPU

v_addc_co_u32 v0, vcc, shared_base, v0, vcc
// CHECK: error: instruction not supported on this GPU

v_addc_co_u32_dpp v255, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_addc_co_u32_e32 v3, vcc, 12345, v3, vcc
// CHECK: error: instruction not supported on this GPU

v_addc_co_u32_e64 v255, s[12:13], v1, v2, s[6:7]
// CHECK: error: instruction not supported on this GPU

v_addc_co_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
// CHECK: error: instruction not supported on this GPU

v_and_or_b32 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_ashr_i32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_ashr_i32_e64 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_ashr_i64 v[254:255], v[1:2], v2
// CHECK: error: instruction not supported on this GPU

v_cmps_eq_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_eq_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_eq_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_eq_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_f_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_f_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_f_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_f_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_ge_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_ge_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_ge_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_ge_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_gt_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_gt_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_gt_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_gt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_le_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_le_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_le_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_le_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_lg_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_lg_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_lg_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_lg_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_lt_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_lt_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_lt_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_lt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_neq_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_neq_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_neq_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_neq_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_nge_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_nge_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_nge_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_nge_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_ngt_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_ngt_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_ngt_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_ngt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_nle_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_nle_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_nle_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_nle_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_nlg_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_nlg_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_nlg_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_nlg_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_nlt_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_nlt_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_nlt_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_nlt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_o_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_o_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_o_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_o_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_tru_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_tru_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_tru_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_tru_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_u_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_u_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmps_u_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmps_u_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_eq_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_eq_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_eq_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_eq_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_f_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_f_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_f_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_f_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_ge_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_ge_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_ge_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_ge_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_gt_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_gt_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_gt_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_gt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_le_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_le_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_le_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_le_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_lg_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_lg_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_lg_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_lg_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_lt_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_lt_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_lt_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_lt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_neq_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_neq_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_neq_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_neq_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nge_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nge_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nge_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nge_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_ngt_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_ngt_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_ngt_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_ngt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nle_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nle_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nle_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nle_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nlg_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nlg_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nlg_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nlg_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nlt_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nlt_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nlt_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_nlt_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_o_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_o_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_o_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_o_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_tru_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_tru_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_tru_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_tru_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_u_f32 vcc, -1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_u_f32_e64 flat_scratch, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cmpsx_u_f64 vcc, -1, v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cmpsx_u_f64_e64 flat_scratch, v[1:2], v[2:3]
// CHECK: error: instruction not supported on this GPU

v_cvt_norm_i16_f16 v5, -4.0
// CHECK: error: instruction not supported on this GPU

v_cvt_norm_i16_f16_dpp v255, v1 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_cvt_norm_i16_f16_e32 v255, v1
// CHECK: error: instruction not supported on this GPU

v_cvt_norm_i16_f16_e64 v255, v1
// CHECK: error: instruction not supported on this GPU

v_cvt_norm_i16_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: error: instruction not supported on this GPU

v_cvt_norm_u16_f16 v5, s101
// CHECK: error: instruction not supported on this GPU

v_cvt_norm_u16_f16_dpp v255, v1 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_cvt_norm_u16_f16_e32 v255, v1
// CHECK: error: instruction not supported on this GPU

v_cvt_norm_u16_f16_e64 v255, v1
// CHECK: error: instruction not supported on this GPU

v_cvt_norm_u16_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: error: instruction not supported on this GPU

v_cvt_pknorm_i16_f16 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_cvt_pknorm_u16_f16 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_div_fixup_legacy_f16 v255, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_div_fixup_legacy_f16_e64 v5, 0.5, v2, v3
// CHECK: error: instruction not supported on this GPU

v_dot2_f32_f16 v0, -v1, -v2, -v3
// CHECK: error: instruction not supported on this GPU

v_dot2_i32_i16 v0, -v1, -v2, -v3
// CHECK: error: instruction not supported on this GPU

v_dot2_u32_u16 v0, -v1, -v2, -v3
// CHECK: error: instruction not supported on this GPU

v_dot2c_f32_f16 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_dot2c_f32_f16_dpp v255, v1, v2  quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_dot2c_f32_f16_e32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_dot2c_i32_i16 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_dot2c_i32_i16_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_dot4_i32_i8 v0, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_dot4_u32_u8 v0, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_dot4c_i32_i8 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_dot4c_i32_i8_dpp v255, v1, v2  quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_dot4c_i32_i8_e32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_dot8_i32_i4 v0, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_dot8_u32_u4 v0, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_dot8c_i32_i4 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_dot8c_i32_i4_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_fma_legacy_f16 v255, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_fma_legacy_f16_e64 v5, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_fma_mix_f32 v0, -abs(v1), v2, v3
// CHECK: error: instruction not supported on this GPU

v_fma_mixhi_f16 v0, -v1, abs(v2), -abs(v3)
// CHECK: error: instruction not supported on this GPU

v_fma_mixlo_f16 v0, abs(v1), -v2, abs(v3)
// CHECK: error: instruction not supported on this GPU

v_fmaak_f32 v255, v1, v2, 0x1121
// CHECK: error: instruction not supported on this GPU

v_fmac_f16 v5, 0x1234, v2
// CHECK: error: instruction not supported on this GPU

v_fmac_f16_dpp v5, v1, v2 quad_perm:[3,2,1,0] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_fmac_f16_e32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_fmac_f16_e64 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_fmac_f32 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_fmac_f32_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_fmac_f32_e32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_fmac_f32_e64 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_fmamk_f32 v255, v1, 0x1121, v3
// CHECK: error: instruction not supported on this GPU

v_interp_p2_legacy_f16 v255, v2, attr0.x, v3
// CHECK: error: instruction not supported on this GPU

v_log_clamp_f32 v1, 0.5
// CHECK: error: instruction not supported on this GPU

v_log_clamp_f32_e64 v255, v1
// CHECK: error: instruction not supported on this GPU

v_lshl_add_u32 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_lshl_b32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_lshl_b32_e64 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_lshl_b64 v[254:255], v[1:2], v2
// CHECK: error: instruction not supported on this GPU

v_lshl_or_b32 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_lshr_b32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_lshr_b32_e64 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_lshr_b64 v[254:255], v[1:2], v2
// CHECK: error: instruction not supported on this GPU

v_mac_legacy_f32 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_mac_legacy_f32_e32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_mac_legacy_f32_e64 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_mad_i32_i16 v255, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_mad_legacy_f16 v255, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_mad_legacy_f16_e64 v5, 0.5, v2, v3
// CHECK: error: instruction not supported on this GPU

v_mad_legacy_i16 v255, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_mad_legacy_i16_e64 v5, 0, v2, v3
// CHECK: error: instruction not supported on this GPU

v_mad_legacy_u16 v255, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_mad_legacy_u16_e64 v5, 0, v2, v3
// CHECK: error: instruction not supported on this GPU

v_mad_mix_f32 v0, -abs(v1), v2, v3
// CHECK: error: instruction not supported on this GPU

v_mad_mixhi_f16 v0, -v1, abs(v2), -abs(v3)
// CHECK: error: instruction not supported on this GPU

v_mad_mixlo_f16 v0, abs(v1), -v2, abs(v3)
// CHECK: error: instruction not supported on this GPU

v_mad_u32_u16 v255, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_max3_f16 v0, src_lds_direct, v0, v0
// CHECK: error: instruction not supported on this GPU

v_max3_i16 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_max3_u16 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_max_legacy_f32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_max_legacy_f32_e64 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_med3_f16 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_med3_i16 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_med3_u16 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_16x16x16f16 a[0:3], a[0:1], a[1:2], -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_16x16x1f32 a[0:15], a0, a1, -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_16x16x2bf16 a[0:15], a0, a1, -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f16 a[0:15], a[0:1], a[1:2], -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_16x16x4f32 a[0:3], a0, a1, -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_16x16x8bf16 a[0:3], a0, a1, -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], 1, v1, a[1:32]
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_32x32x2bf16 a[0:31], a0, a1, -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_32x32x2f32 a[0:15], a0, a1, -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_32x32x4bf16 a[0:15], a0, a1, -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_32x32x4f16 a[0:31], a[0:1], a[1:2], -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[1:2], -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_4x4x1f32 a[0:3], a0, a1, -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_4x4x2bf16 a[0:3], a0, a1, -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_f32_4x4x4f16 a[0:3], a[0:1], a[1:2], -2.0
// CHECK: error: instruction not supported on this GPU

v_mfma_i32_16x16x16i8 a[0:3], a0, a1, 2
// CHECK: error: instruction not supported on this GPU

v_mfma_i32_16x16x4i8 a[0:15], a0, a1, 2
// CHECK: error: instruction not supported on this GPU

v_mfma_i32_32x32x4i8 a[0:31], a0, a1, 2
// CHECK: error: instruction not supported on this GPU

v_mfma_i32_32x32x8i8 a[0:15], a0, a1, 2
// CHECK: error: instruction not supported on this GPU

v_mfma_i32_4x4x4i8 a[0:3], a0, a1, 2
// CHECK: error: instruction not supported on this GPU

v_min3_f16 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_min3_i16 v0, src_lds_direct, v0, v0
// CHECK: error: instruction not supported on this GPU

v_min3_u16 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_min_legacy_f32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_min_legacy_f32_e64 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_movrelsd_2_b32 v0, v255 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: error: instruction not supported on this GPU

v_movrelsd_2_b32_dpp v0, v2 quad_perm:[3,2,1,0] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_movrelsd_2_b32_e32 v5, 1
// CHECK: error: instruction not supported on this GPU

v_movrelsd_2_b32_e64 v255, v1
// CHECK: error: instruction not supported on this GPU

v_movrelsd_2_b32_sdwa v0, 0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: error: instruction not supported on this GPU

v_mullit_f32 v255, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_or3_b32 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_pack_b32_f16 v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_permlane16_b32 v0, lds_direct, s0, s0
// CHECK: error: instruction not supported on this GPU

v_permlanex16_b32 v0, lds_direct, s0, s0
// CHECK: error: instruction not supported on this GPU

v_pipeflush
// CHECK: error: instruction not supported on this GPU

v_pipeflush_e64
// CHECK: error: instruction not supported on this GPU

v_pk_add_f16 v0, execz, v0
// CHECK: error: instruction not supported on this GPU

v_pk_add_i16 v0, src_lds_direct, v0
// CHECK: error: instruction not supported on this GPU

v_pk_add_u16 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_pk_ashrrev_i16 v0, lds_direct, v0
// CHECK: error: instruction not supported on this GPU

v_pk_fma_f16 v0, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_pk_fmac_f16 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_pk_lshlrev_b16 v0, lds_direct, v0
// CHECK: error: instruction not supported on this GPU

v_pk_lshrrev_b16 v0, lds_direct, v0
// CHECK: error: instruction not supported on this GPU

v_pk_mad_i16 v0, src_lds_direct, v0, v0
// CHECK: error: instruction not supported on this GPU

v_pk_mad_u16 v255, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

v_pk_max_f16 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_pk_max_i16 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_pk_max_u16 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_pk_min_f16 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_pk_min_i16 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_pk_min_u16 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_pk_mul_f16 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_pk_mul_lo_u16 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_pk_sub_i16 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_pk_sub_u16 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_rcp_clamp_f32 v255, v1
// CHECK: error: instruction not supported on this GPU

v_rcp_clamp_f32_e64 v255, v1
// CHECK: error: instruction not supported on this GPU

v_rcp_clamp_f64 v[254:255], v[1:2]
// CHECK: error: instruction not supported on this GPU

v_rcp_clamp_f64_e64 v[254:255], v[1:2]
// CHECK: error: instruction not supported on this GPU

v_rcp_legacy_f32 v255, v1
// CHECK: error: instruction not supported on this GPU

v_rcp_legacy_f32_e64 v255, v1
// CHECK: error: instruction not supported on this GPU

v_rsq_clamp_f32 v255, v1
// CHECK: error: instruction not supported on this GPU

v_rsq_clamp_f32_e64 v255, v1
// CHECK: error: instruction not supported on this GPU

v_rsq_clamp_f64 v[254:255], v[1:2]
// CHECK: error: instruction not supported on this GPU

v_rsq_clamp_f64_e64 v[254:255], v[1:2]
// CHECK: error: instruction not supported on this GPU

v_rsq_legacy_f32 v255, v1
// CHECK: error: instruction not supported on this GPU

v_rsq_legacy_f32_e64 v255, v1
// CHECK: error: instruction not supported on this GPU

v_sat_pk_u8_i16 v255, v1
// CHECK: error: instruction not supported on this GPU

v_sat_pk_u8_i16_dpp v5, v1 quad_perm:[3,2,1,0] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_sat_pk_u8_i16_e64 v5, -1
// CHECK: error: instruction not supported on this GPU

v_sat_pk_u8_i16_sdwa v5, sext(v1) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: error: instruction not supported on this GPU

v_screen_partition_4se_b32 v5, -1
// CHECK: error: instruction not supported on this GPU

v_screen_partition_4se_b32_dpp v5, v1 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 bound_ctrl:0
// CHECK: error: instruction not supported on this GPU

v_screen_partition_4se_b32_e64 v5, -1
// CHECK: error: instruction not supported on this GPU

v_screen_partition_4se_b32_sdwa v5, v1 src0_sel:BYTE_0
// CHECK: error: instruction not supported on this GPU

v_sub_co_ci_u32_dpp v0, vcc, v0, v0, vcc dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: error: instruction not supported on this GPU

v_sub_co_ci_u32_e32 v255, vcc, v1, v2, vcc
// CHECK: error: instruction not supported on this GPU

v_sub_co_ci_u32_e64 v255, s12, v1, v2, s6
// CHECK: error: instruction not supported on this GPU

v_sub_co_ci_u32_sdwa v1, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// CHECK: error: instruction not supported on this GPU

v_sub_co_u32 v0, s0, v0, v2
// CHECK: error: instruction not supported on this GPU

v_sub_co_u32_dpp v255, vcc, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_sub_co_u32_e32 v2, vcc, s0, v2
// CHECK: error: instruction not supported on this GPU

v_sub_co_u32_e64 v0, s0, v0, v2
// CHECK: error: instruction not supported on this GPU

v_sub_co_u32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// CHECK: error: instruction not supported on this GPU

v_sub_i16 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_sub_i32 v1, s[0:1], v2, v3
// CHECK: error: instruction not supported on this GPU

v_sub_i32_e64 v255, s[12:13], v1, v2
// CHECK: error: instruction not supported on this GPU

v_sub_nc_i16 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_sub_nc_i32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_sub_nc_u16 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_sub_nc_u32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// CHECK: error: instruction not supported on this GPU

v_sub_nc_u32_e32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_sub_nc_u32_e64 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_sub_nc_u32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: error: instruction not supported on this GPU

v_subb_co_u32 v1, vcc, v2, v3, vcc row_shl:1 row_mask:0xa bank_mask:0x1 bound_ctrl:0
// CHECK: error: instruction not supported on this GPU

v_subb_co_u32_dpp v255, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_subb_co_u32_e64 v255, s[12:13], v1, v2, s[6:7]
// CHECK: error: instruction not supported on this GPU

v_subb_co_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
// CHECK: error: instruction not supported on this GPU

v_subbrev_co_u32 v0, vcc, src_lds_direct, v0, vcc
// CHECK: error: instruction not supported on this GPU

v_subbrev_co_u32_dpp v255, vcc, v1, v2, vcc quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_subbrev_co_u32_e64 v255, s[12:13], v1, v2, s[6:7]
// CHECK: error: instruction not supported on this GPU

v_subbrev_co_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
// CHECK: error: instruction not supported on this GPU

v_subrev_co_ci_u32 v0, vcc_lo, src_lds_direct, v0, vcc_lo
// CHECK: error: instruction not supported on this GPU

v_subrev_co_ci_u32_dpp v0, vcc, v0, v0, vcc dpp8:[7,6,5,4,3,2,1,0]
// CHECK: error: instruction not supported on this GPU

v_subrev_co_ci_u32_e32 v1, 0, v1
// CHECK: error: instruction not supported on this GPU

v_subrev_co_ci_u32_e64 v255, s12, v1, v2, s6
// CHECK: error: instruction not supported on this GPU

v_subrev_co_ci_u32_sdwa v1, v1, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// CHECK: error: instruction not supported on this GPU

v_subrev_co_u32 v0, s0, src_lds_direct, v0
// CHECK: error: instruction not supported on this GPU

v_subrev_co_u32_dpp v255, vcc, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_subrev_co_u32_e32 v2, vcc, s0, v2
// CHECK: error: instruction not supported on this GPU

v_subrev_co_u32_e64 v0, s0, v0, v2
// CHECK: error: instruction not supported on this GPU

v_subrev_co_u32_sdwa v0, v0, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// CHECK: error: instruction not supported on this GPU

v_subrev_i32 v1, s[0:1], v2, v3
// CHECK: error: instruction not supported on this GPU

v_subrev_i32_e64 v255, s[12:13], v1, v2
// CHECK: error: instruction not supported on this GPU

v_subrev_nc_u32 v0, src_lds_direct, v0
// CHECK: error: instruction not supported on this GPU

v_subrev_nc_u32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] fi:1
// CHECK: error: instruction not supported on this GPU

v_subrev_nc_u32_e32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_subrev_nc_u32_e64 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_subrev_nc_u32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: error: instruction not supported on this GPU

v_swap_b32 v1, 1
// CHECK: error: instruction not supported on this GPU

v_swap_b32_e32 v1, v2
// CHECK: error: instruction not supported on this GPU

v_swaprel_b32 v255, v1
// CHECK: error: instruction not supported on this GPU

v_xad_u32 v1, v2, v3, v4
// CHECK: error: instruction not supported on this GPU

v_xnor_b32 v0, v1, v2
// CHECK: error: instruction not supported on this GPU

v_xnor_b32_dpp v255, v1, v2  quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: error: instruction not supported on this GPU

v_xnor_b32_e32 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_xnor_b32_e64 v255, v1, v2
// CHECK: error: instruction not supported on this GPU

v_xnor_b32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: error: instruction not supported on this GPU

v_xor3_b32 v255, v1, v2, v3
// CHECK: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// Unsupported e32 variants.
//===----------------------------------------------------------------------===//

v_cvt_pkrtz_f16_f32_e32 v255, v1, v2
// CHECK: error: e32 variant of this instruction is not supported

//===----------------------------------------------------------------------===//
// Unsupported dpp variants.
//===----------------------------------------------------------------------===//

v_movreld_b32_dpp v1, v0 quad_perm:[3,2,1,0] row_mask:0x0 bank_mask:0x0
// CHECK: error: dpp variant of this instruction is not supported

v_movrels_b32_dpp v1, v0 quad_perm:[3,2,1,0] row_mask:0x0 bank_mask:0x0 fi:1
// CHECK: error: dpp variant of this instruction is not supported

v_movrelsd_b32_dpp v0, v255 quad_perm:[3,2,1,0] row_mask:0x0 bank_mask:0x0
// CHECK: error: dpp variant of this instruction is not supported

//===----------------------------------------------------------------------===//
// Unsupported sdwa variants.
//===----------------------------------------------------------------------===//

v_movreld_b32_sdwa v0, 64 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: error: sdwa variant of this instruction is not supported

v_movrels_b32_sdwa v0, 1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: error: sdwa variant of this instruction is not supported

v_movrelsd_b32_sdwa v0, 1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: error: sdwa variant of this instruction is not supported
