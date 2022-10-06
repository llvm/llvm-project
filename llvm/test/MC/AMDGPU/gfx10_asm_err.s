// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx601 %s 2>&1 | FileCheck --check-prefixes=GFX6-7,GFX6-8,GFX6-9 --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx701 %s 2>&1 | FileCheck --check-prefixes=GFX6-7,GFX6-8,GFX6-9 --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx801 %s 2>&1 | FileCheck --check-prefixes=GFX6-8,GFX6-9,GFX8-9 --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck --check-prefixes=GFX6-9,GFX8-9 --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=GFX10 --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=GFX10 --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// ENC_DS.
//===----------------------------------------------------------------------===//

// GFX9+.

ds_write_b8_d16_hi v1, v2
// GFX6-8: error: instruction not supported on this GPU

ds_write_b16_d16_hi v1, v2
// GFX6-8: error: instruction not supported on this GPU

ds_read_u8_d16 v5, v1
// GFX6-8: error: instruction not supported on this GPU

ds_read_u8_d16_hi v5, v1
// GFX6-8: error: instruction not supported on this GPU

ds_read_i8_d16 v5, v1
// GFX6-8: error: instruction not supported on this GPU

ds_read_i8_d16_hi v5, v1
// GFX6-8: error: instruction not supported on this GPU

ds_read_u16_d16 v5, v1
// GFX6-8: error: instruction not supported on this GPU

ds_read_u16_d16_hi v5, v1
// GFX6-8: error: instruction not supported on this GPU

ds_write_addtid_b32 v5
// GFX6-8: error: instruction not supported on this GPU

ds_read_addtid_b32 v5
// GFX6-8: error: instruction not supported on this GPU

// GFX8+.

ds_add_src2_f32 v1
// GFX6-7: error: instruction not supported on this GPU

ds_add_f32 v0, v1
// GFX6-7: error: instruction not supported on this GPU

ds_add_rtn_f32 v0, v1, v2
// GFX6-7: error: instruction not supported on this GPU

ds_permute_b32 v0, v1, v2
// GFX6-7: error: instruction not supported on this GPU

ds_bpermute_b32 v0, v1, v2
// GFX6-7: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// ENC_SOP1.
//===----------------------------------------------------------------------===//

// GFX10+.

s_and_saveexec_b32 s0, s1
// GFX6-9: error: instruction not supported on this GPU

s_or_saveexec_b32 s0, s1
// GFX6-9: error: instruction not supported on this GPU

s_xor_saveexec_b32 s0, s1
// GFX6-9: error: instruction not supported on this GPU

s_andn2_saveexec_b32 s0, s1
// GFX6-9: error: instruction not supported on this GPU

s_orn2_saveexec_b32 s0, s1
// GFX6-9: error: instruction not supported on this GPU

s_nand_saveexec_b32 s0, s1
// GFX6-9: error: instruction not supported on this GPU

s_nor_saveexec_b32 s0, s1
// GFX6-9: error: instruction not supported on this GPU

s_xnor_saveexec_b32 s0, s1
// GFX6-9: error: instruction not supported on this GPU

s_andn1_saveexec_b32 s0, s1
// GFX6-9: error: instruction not supported on this GPU

s_orn1_saveexec_b32 s0, s1
// GFX6-9: error: instruction not supported on this GPU

s_andn1_wrexec_b32 s0, s1
// GFX6-9: error: instruction not supported on this GPU

s_andn2_wrexec_b32 s0, s1
// GFX6-9: error: instruction not supported on this GPU

s_movrelsd_2_b32 s0, s1
// GFX6-9: error: instruction not supported on this GPU

// GFX9+.

s_andn1_saveexec_b64 s[0:1], s[2:3]
// GFX6-8: error: instruction not supported on this GPU

s_orn1_saveexec_b64 s[0:1], s[2:3]
// GFX6-8: error: instruction not supported on this GPU

s_andn1_wrexec_b64 s[0:1], s[2:3]
// GFX6-8: error: instruction not supported on this GPU

s_andn2_wrexec_b64 s[0:1], s[2:3]
// GFX6-8: error: instruction not supported on this GPU

s_bitreplicate_b64_b32 s[0:1], s2
// GFX6-8: error: instruction not supported on this GPU

// GFX8, GFX9.

s_set_gpr_idx_idx s0
// GFX10: error: instruction not supported on this GPU
// GFX6-7: error: instruction not supported on this GPU

// GFX6, GFX7, GFX8, GFX9.

s_cbranch_join s0
// GFX10: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// ENC_SOP2.
//===----------------------------------------------------------------------===//

// GFX9+.

s_lshl1_add_u32 s0, s1, s2
// GFX6-8: error: instruction not supported on this GPU

s_lshl2_add_u32 s0, s1, s2
// GFX6-8: error: instruction not supported on this GPU

s_lshl3_add_u32 s0, s1, s2
// GFX6-8: error: instruction not supported on this GPU

s_lshl4_add_u32 s0, s1, s2
// GFX6-8: error: instruction not supported on this GPU

s_mul_hi_u32 s0, s1, s2
// GFX6-8: error: instruction not supported on this GPU

s_mul_hi_i32 s0, s1, s2
// GFX6-8: error: instruction not supported on this GPU

s_pack_ll_b32_b16 s0, s1, s2
// GFX6-8: error: instruction not supported on this GPU

s_pack_lh_b32_b16 s0, s1, s2
// GFX6-8: error: instruction not supported on this GPU

s_pack_hh_b32_b16 s0, s1, s2
// GFX6-8: error: instruction not supported on this GPU

// GFX8, GFX9.

s_rfe_restore_b64 s[0:1], s2
// GFX10: error: instruction not supported on this GPU
// GFX6-7: error: instruction not supported on this GPU

// GFX6, GFX7, GFX8, GFX9.

s_cbranch_g_fork s[0:1], s[2:3]
// GFX10: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// ENC_SOPC.
//===----------------------------------------------------------------------===//

// GFX8+.

s_cmp_eq_u64 s[0:1], s[2:3]
// GFX6-7: error: instruction not supported on this GPU

s_cmp_lg_u64 s[0:1], s[2:3]
// GFX6-7: error: instruction not supported on this GPU

// GFX6, GFX7, GFX8, GFX9.

s_setvskip s0, s1
// GFX10: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// ENC_SOPK.
//===----------------------------------------------------------------------===//

// GFX10+.

s_version 0x1234
// GFX6-9: error: instruction not supported on this GPU

s_waitcnt_vscnt s0, 0x1234
// GFX6-9: error: instruction not supported on this GPU

s_waitcnt_vmcnt s0, 0x1234
// GFX6-9: error: instruction not supported on this GPU

s_waitcnt_expcnt s0, 0x1234
// GFX6-9: error: instruction not supported on this GPU

s_waitcnt_lgkmcnt s0, 0x1234
// GFX6-9: error: instruction not supported on this GPU

s_subvector_loop_begin s0, 0x1234
// GFX6-9: error: instruction not supported on this GPU

s_subvector_loop_end s0, 0x1234
// GFX6-9: error: instruction not supported on this GPU

// GFX9+.

s_call_b64 s[0:1], 0x1234
// GFX6-8: error: instruction not supported on this GPU

// GFX6, GFX7, GFX8, GFX9.

s_cbranch_i_fork s[0:1], 0x1234
// GFX10: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// ENC_SOPP.
//===----------------------------------------------------------------------===//

// GFX10+.

s_code_end
// GFX6-9: error: instruction not supported on this GPU

s_inst_prefetch 0x0
// GFX6-9: error: instruction not supported on this GPU

s_clause 0x0
// GFX6-9: error: instruction not supported on this GPU

s_round_mode 0x0
// GFX6-9: error: instruction not supported on this GPU

s_denorm_mode 0x0
// GFX6-9: error: instruction not supported on this GPU

s_ttracedata_imm 0x0
// GFX6-9: error: instruction not supported on this GPU

// GFX9+.

s_endpgm_ordered_ps_done
// GFX6-8: error: instruction not supported on this GPU

// GFX8+.

s_wakeup
// GFX6-7: error: instruction not supported on this GPU

s_endpgm_saved
// GFX6-7: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// DPP8.
//===----------------------------------------------------------------------===//

v_mov_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX6-7: error: dpp variant of this instruction is not supported
// GFX8-9: error: not a valid operand

//===----------------------------------------------------------------------===//
// VOP2
//===----------------------------------------------------------------------===//

v_fmaak_f32 v0, 0xff32ff, v0, 0x11213141
// GFX6-9: error: instruction not supported on this GPU
// GFX10: error: only one unique literal operand is allowed

v_fmamk_f32 v0, 0xff32ff, 0x11213141, v0
// GFX6-9: error: instruction not supported on this GPU
// GFX10: error: only one unique literal operand is allowed

v_fmaak_f32 v0, 0xff32, v0, 0x1122
// GFX6-9: error: instruction not supported on this GPU
// GFX10: error: only one unique literal operand is allowed

v_fmamk_f32 v0, 0xff32, 0x1122, v0
// GFX6-9: error: instruction not supported on this GPU
// GFX10: error: only one unique literal operand is allowed

//===----------------------------------------------------------------------===//
// VOP2 E64.
//===----------------------------------------------------------------------===//

v_add_co_ci_u32 v5, 0, v1, v2, vcc
// GFX6-7: error: instruction not supported on this GPU
// GFX8-9: error: instruction not supported on this GPU
// GFX10: error: invalid operand for instruction

v_add_co_ci_u32 v5, vcc, v1, v2, 0
// GFX6-7: error: instruction not supported on this GPU
// GFX8-9: error: instruction not supported on this GPU
// GFX10: error: invalid operand for instruction

v_add_co_ci_u32 v5, 0, v1, v2, vcc_lo
// GFX6-7: error: instruction not supported on this GPU
// GFX8-9: error: instruction not supported on this GPU
// GFX10: error: invalid operand for instruction

v_add_co_ci_u32 v5, vcc_lo, v1, v2, 0
// GFX6-7: error: instruction not supported on this GPU
// GFX8-9: error: instruction not supported on this GPU
// GFX10: error: invalid operand for instruction
