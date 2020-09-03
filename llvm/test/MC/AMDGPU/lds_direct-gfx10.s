// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck %s --check-prefix=GFX10
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck %s --check-prefix=NOGFX10 --implicit-check-not=error:

v_readfirstlane_b32 s0, lds_direct
// GFX10: v_readfirstlane_b32 s0, src_lds_direct ; encoding: [0xfe,0x04,0x00,0x7e]

v_readlane_b32 s0, lds_direct, s0
// GFX10: v_readlane_b32 s0, src_lds_direct, s0 ; encoding: [0x00,0x00,0x60,0xd7,0xfe,0x00,0x00,0x00]

v_writelane_b32 v0, lds_direct, s0
// GFX10: v_writelane_b32 v0, src_lds_direct, s0 ; encoding: [0x00,0x00,0x61,0xd7,0xfe,0x00,0x00,0x00]

v_permlane16_b32 v0, lds_direct, s0, s0
// NOGFX10: error: invalid operand for instruction

v_permlanex16_b32 v0, lds_direct, s0, s0
// NOGFX10: error: invalid operand for instruction

v_ashrrev_i16 v0, src_lds_direct, v0
// NOGFX10: error: invalid use of lds_direct

v_ashrrev_i32 v0, src_lds_direct, v0
// NOGFX10: error: invalid use of lds_direct

v_lshlrev_b16 v0, src_lds_direct, v0
// NOGFX10: error: invalid use of lds_direct

v_lshlrev_b32 v0, src_lds_direct, v0
// NOGFX10: error: invalid use of lds_direct

v_lshrrev_b16 v0, src_lds_direct, v0
// NOGFX10: error: invalid use of lds_direct

v_lshrrev_b32 v0, src_lds_direct, v0
// NOGFX10: error: invalid use of lds_direct

v_pk_ashrrev_i16 v0, src_lds_direct, v0
// NOGFX10: error: invalid use of lds_direct

v_pk_lshlrev_b16 v0, src_lds_direct, v0
// NOGFX10: error: invalid use of lds_direct

v_pk_lshrrev_b16 v0, src_lds_direct, v0
// NOGFX10: error: invalid use of lds_direct

v_subrev_co_ci_u32 v0, vcc_lo, src_lds_direct, v0, vcc_lo
// NOGFX10: error: invalid use of lds_direct

v_subrev_co_u32 v0, s0, src_lds_direct, v0
// NOGFX10: error: invalid use of lds_direct

v_subrev_f16 v0, src_lds_direct, v0
// NOGFX10: error: invalid use of lds_direct

v_subrev_f32 v0, src_lds_direct, v0
// NOGFX10: error: invalid use of lds_direct

v_subrev_nc_u32 v0, src_lds_direct, v0
// NOGFX10: error: invalid use of lds_direct
