// RUN: not llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=CIVI --check-prefix=VI

// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s 2>&1 | FileCheck %s -check-prefix=NOVI

//===----------------------------------------------------------------------===//
// Generic Checks for floating-point instructions (These have modifiers).
//===----------------------------------------------------------------------===//

// TODO: 64-bit encoding of instructions with modifiers

// _e32 suffix
// SICI: v_add_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x06]
v_add_f32_e32 v1, v2, v3

// src0 inline immediate
// SICI: v_add_f32_e32 v1, 1.0, v3 ; encoding: [0xf2,0x06,0x02,0x06]
v_add_f32 v1, 1.0, v3

// src0 negative inline immediate
// SICI: v_add_f32_e32 v1, -1.0, v3 ; encoding: [0xf3,0x06,0x02,0x06]
v_add_f32 v1, -1.0, v3

// src0 literal
// SICI: v_add_f32_e32 v1, 0x42c80000, v3 ; encoding: [0xff,0x06,0x02,0x06,0x00,0x00,0xc8,0x42]
v_add_f32 v1, 100.0, v3

// src0 negative literal
// SICI: v_add_f32_e32 v1, 0xc2c80000, v3 ; encoding: [0xff,0x06,0x02,0x06,0x00,0x00,0xc8,0xc2]
v_add_f32 v1, -100.0, v3

//===----------------------------------------------------------------------===//
// Generic Checks for integer instructions (These don't have modifiers).
//===----------------------------------------------------------------------===//

// _e32 suffix
// SICI: v_mul_i32_i24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x12]
v_mul_i32_i24_e32 v1, v2, v3

// _e64 suffix
// SICI: v_mul_i32_i24_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x07,0x02,0x00]
v_mul_i32_i24_e64 v1, v2, v3

// src0 inline
// SICI: v_mul_i32_i24_e32 v1, 3, v3 ; encoding: [0x83,0x06,0x02,0x12]
v_mul_i32_i24_e32 v1, 3, v3

// src0 negative inline
// SICI: v_mul_i32_i24_e32 v1, -3, v3 ; encoding: [0xc3,0x06,0x02,0x12]
v_mul_i32_i24_e32 v1, -3, v3

// src1 inline
// SICI: v_mul_i32_i24_e64 v1, v2, 3 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x07,0x01,0x00]
v_mul_i32_i24_e64 v1, v2, 3

// src1 negative inline
// SICI: v_mul_i32_i24_e64 v1, v2, -3 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x87,0x01,0x00]
v_mul_i32_i24_e64 v1, v2, -3

// src0 literal
// SICI: v_mul_i32_i24_e32 v1, 0x64, v3 ; encoding: [0xff,0x06,0x02,0x12,0x64,0x00,0x00,0x00]
v_mul_i32_i24_e32 v1, 100, v3

// src1 negative literal
// SICI: v_mul_i32_i24_e32 v1, 0xffffff9c, v3 ; encoding: [0xff,0x06,0x02,0x12,0x9c,0xff,0xff,0xff]
v_mul_i32_i24_e32 v1, -100, v3

//===----------------------------------------------------------------------===//
// Checks for legal operands
//===----------------------------------------------------------------------===//

// src0 sgpr
// SICI: v_mul_i32_i24_e32 v1, s2, v3 ; encoding: [0x02,0x06,0x02,0x12]
v_mul_i32_i24_e32 v1, s2, v3

// src1 sgpr
// SICI: v_mul_i32_i24_e64 v1, v2, s3 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x07,0x00,0x00]
v_mul_i32_i24_e64 v1, v2, s3

// src0, src1 same sgpr
// SICI: v_mul_i32_i24_e64 v1, s2, s2 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x04,0x00,0x00]
v_mul_i32_i24_e64 v1, s2, s2

// src0 sgpr, src1 inline
// SICI: v_mul_i32_i24_e64 v1, s2, 3 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x06,0x01,0x00]
v_mul_i32_i24_e64 v1, s2, 3

// src0 inline src1 sgpr
// SICI: v_mul_i32_i24_e64 v1, 3, s3 ; encoding: [0x01,0x00,0x12,0xd2,0x83,0x06,0x00,0x00]
v_mul_i32_i24_e64 v1, 3, s3

// SICI: v_add_i32_e32 v0, vcc, 0.5, v0 ; encoding: [0xf0,0x00,0x00,0x4a]
// VI: v_add_i32_e32 v0, vcc, 0.5, v0 ; encoding: [0xf0,0x00,0x00,0x32]
v_add_i32_e32 v0, vcc, 0.5, v0

// SICI: v_add_i32_e32 v0, vcc, 0x40480000, v0 ; encoding: [0xff,0x00,0x00,0x4a,0x00,0x00,0x48,0x40]
// VI: v_add_i32_e32 v0, vcc, 0x40480000, v0 ; encoding: [0xff,0x00,0x00,0x32,0x00,0x00,0x48,0x40]
v_add_i32_e32 v0, vcc, 3.125, v0

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

// GCN: v_cndmask_b32_e32 v1, v2, v3, vcc ; encoding: [0x02,0x07,0x02,0x00]
v_cndmask_b32 v1, v2, v3, vcc

// GCN: v_cndmask_b32_e32 v1, v2, v3, vcc ; encoding: [0x02,0x07,0x02,0x00]
v_cndmask_b32_e32 v1, v2, v3, vcc

// SICI: v_readlane_b32 s1, v2, s3 ; encoding: [0x02,0x07,0x02,0x02]
// VI:   v_readlane_b32 s1, v2, s3 ; encoding: [0x01,0x00,0x89,0xd2,0x02,0x07,0x00,0x00]
v_readlane_b32 s1, v2, s3

// SICI: v_writelane_b32 v1, s2, 4 ; encoding: [0x02,0x08,0x03,0x04]
// VI:   v_writelane_b32 v1, s2, 4 ; encoding: [0x01,0x00,0x8a,0xd2,0x02,0x08,0x01,0x00]
v_writelane_b32 v1, s2, 4

// SICI: v_add_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x06]
// VI:   v_add_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x02]
v_add_f32 v1, v2, v3

// SICI: v_sub_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x08]
// VI:   v_sub_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x04]
v_sub_f32 v1, v2, v3

// SICI: v_subrev_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x0a]
// VI:   v_subrev_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x06]
v_subrev_f32 v1, v2, v3

// SICI: v_mac_legacy_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x0c]
// NOVI: error: instruction not supported on this GPU
// NOVI: v_mac_legacy_f32 v1, v2, v3
v_mac_legacy_f32 v1, v2, v3

// SICI: v_mul_legacy_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x0e]
// VI:   v_mul_legacy_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x08]
v_mul_legacy_f32_e32 v1, v2, v3

// SICI: v_mul_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x10]
// VI:   v_mul_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x0a]
v_mul_f32 v1, v2, v3

// SICI: v_mul_i32_i24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x12]
// VI:   v_mul_i32_i24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x0c]
v_mul_i32_i24_e32 v1, v2, v3

// SICI: v_mul_hi_i32_i24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x14]
// VI:   v_mul_hi_i32_i24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x0e]
v_mul_hi_i32_i24_e32 v1, v2, v3

// SICI: v_mul_u32_u24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x16]
// VI:   v_mul_u32_u24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x10]
v_mul_u32_u24_e32 v1, v2, v3

// SICI: v_mul_hi_u32_u24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x18]
// VI:   v_mul_hi_u32_u24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x12]
v_mul_hi_u32_u24_e32 v1, v2, v3

// SICI: v_min_legacy_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x1a]
// NOVI: error: instruction not supported on this GPU
// NOVI: v_min_legacy_f32_e32 v1, v2, v3
v_min_legacy_f32_e32 v1, v2, v3

// SICI: v_max_legacy_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x1c]
// NOVI: error: instruction not supported on this GPU
// NOVI: v_max_legacy_f32 v1, v2, v3
v_max_legacy_f32 v1, v2, v3

// SICI: v_min_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x1e]
// VI:   v_min_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x14]
v_min_f32_e32 v1, v2, v3

// SICI: v_max_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x20]
// VI:   v_max_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x16]
v_max_f32 v1, v2 v3

// SICI: v_min_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x22]
// VI:   v_min_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x18]
v_min_i32_e32 v1, v2, v3

// SICI: v_max_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x24]
// VI:   v_max_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x1a]
v_max_i32_e32 v1, v2, v3

// SICI: v_min_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x26]
// VI:   v_min_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x1c]
v_min_u32_e32 v1, v2, v3

// SICI: v_max_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x28]
// VI:   v_max_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x1e]
v_max_u32_e32 v1, v2, v3

// SICI: v_lshr_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x2a]
// NOVI: error: instruction not supported on this GPU
// NOVI: v_lshr_b32_e32 v1, v2, v3
v_lshr_b32_e32 v1, v2, v3

// SICI: v_lshrrev_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x2c]
// VI:   v_lshrrev_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x20]
v_lshrrev_b32_e32 v1, v2, v3

// SICI: v_ashr_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x2e]
// NOVI: error: instruction not supported on this GPU
// NOVI: v_ashr_i32_e32 v1, v2, v3
v_ashr_i32_e32 v1, v2, v3

// SICI: v_ashrrev_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x30]
// VI:   v_ashrrev_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x22]
v_ashrrev_i32_e32 v1, v2, v3

// SICI: v_lshl_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x32]
// NOVI: error: instruction not supported on this GPU
// NOVI: v_lshl_b32_e32 v1, v2, v3
v_lshl_b32_e32 v1, v2, v3

// SICI: v_lshlrev_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x34]
// VI:   v_lshlrev_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x24]
v_lshlrev_b32_e32 v1, v2, v3

// SICI: v_and_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x36]
// VI:   v_and_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x26]
v_and_b32_e32 v1, v2, v3

// SICI: v_or_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x38]
// VI:   v_or_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x28]
v_or_b32_e32 v1, v2, v3

// SICI: v_xor_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x3a]
// VI:   v_xor_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x2a]
v_xor_b32_e32 v1, v2, v3

// SICI: v_bfm_b32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x3c,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_bfm_b32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x93,0xd2,0x02,0x07,0x02,0x00]
v_bfm_b32_e64 v1, v2, v3

// SICI: v_mac_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x3e]
// VI:   v_mac_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x2c]
v_mac_f32_e32 v1, v2, v3

// SICI: v_madmk_f32_e32 v1, v2, 0x42800000, v3 ; encoding: [0x02,0x07,0x02,0x40,0x00,0x00,0x80,0x42]
// VI:   v_madmk_f32_e32 v1, v2, 0x42800000, v3 ; encoding: [0x02,0x07,0x02,0x2e,0x00,0x00,0x80,0x42]
v_madmk_f32_e32 v1, v2, 64.0, v3

// SICI: v_madak_f32_e32 v1, v2, v3, 0x42800000 ; encoding: [0x02,0x07,0x02,0x42,0x00,0x00,0x80,0x42]
// VI:   v_madak_f32_e32 v1, v2, v3, 0x42800000 ; encoding: [0x02,0x07,0x02,0x30,0x00,0x00,0x80,0x42]
v_madak_f32_e32 v1, v2, v3, 64.0

// SICI: v_bcnt_u32_b32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x44,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_bcnt_u32_b32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x8b,0xd2,0x02,0x07,0x02,0x00]
v_bcnt_u32_b32_e64 v1, v2, v3

// SICI: v_mbcnt_lo_u32_b32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x46,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_mbcnt_lo_u32_b32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x8c,0xd2,0x02,0x07,0x02,0x00]
v_mbcnt_lo_u32_b32_e64 v1, v2, v3

// SICI: v_mbcnt_hi_u32_b32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x48,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_mbcnt_hi_u32_b32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x8d,0xd2,0x02,0x07,0x02,0x00]
v_mbcnt_hi_u32_b32_e64 v1, v2, v3

// SICI: v_add_i32_e32 v1, vcc, v2, v3 ; encoding: [0x02,0x07,0x02,0x4a]
// VI:   v_add_i32_e32 v1, vcc, v2, v3 ; encoding: [0x02,0x07,0x02,0x32]
v_add_i32_e32 v1, vcc, v2, v3

// SICI: v_add_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x4a,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_add_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x19,0xd1,0x02,0x07,0x02,0x00]
v_add_i32 v1, s[0:1], v2, v3

// SICI: v_add_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x4a,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_add_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x19,0xd1,0x02,0x07,0x02,0x00]
v_add_i32_e64 v1, s[0:1], v2, v3

// SICI: v_add_i32_e64 v1, vcc, v2, v3 ; encoding: [0x01,0x6a,0x4a,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_add_i32_e64 v1, vcc, v2, v3 ; encoding: [0x01,0x6a,0x19,0xd1,0x02,0x07,0x02,0x00]
v_add_i32_e64 v1, vcc, v2, v3

// SICI: v_add_i32_e32 v1, vcc, v2, v3 ; encoding: [0x02,0x07,0x02,0x4a]
// VI:   v_add_i32_e32 v1, vcc, v2, v3 ; encoding: [0x02,0x07,0x02,0x32]
v_add_u32 v1, vcc, v2, v3

// SICI: v_add_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x4a,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_add_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x19,0xd1,0x02,0x07,0x02,0x00]
v_add_u32 v1, s[0:1], v2, v3

// SICI: v_sub_i32_e32 v1, vcc, v2, v3 ; encoding: [0x02,0x07,0x02,0x4c]
// VI:   v_sub_i32_e32 v1, vcc, v2, v3 ; encoding: [0x02,0x07,0x02,0x34]
v_sub_i32 v1, vcc, v2, v3

// SICI: v_sub_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x4c,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_sub_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x1a,0xd1,0x02,0x07,0x02,0x00]
v_sub_i32 v1, s[0:1], v2, v3

// SICI: v_sub_i32_e32 v1, vcc, v2, v3 ; encoding: [0x02,0x07,0x02,0x4c]
// VI:   v_sub_i32_e32 v1, vcc, v2, v3 ; encoding: [0x02,0x07,0x02,0x34]
v_sub_u32 v1, vcc, v2, v3

// SICI: v_sub_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x4c,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_sub_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x1a,0xd1,0x02,0x07,0x02,0x00]
v_sub_u32 v1, s[0:1], v2, v3

// SICI: v_subrev_i32_e32 v1, vcc, v2, v3 ; encoding: [0x02,0x07,0x02,0x4e]
// VI:   v_subrev_i32_e32 v1, vcc, v2, v3 ; encoding: [0x02,0x07,0x02,0x36]
v_subrev_i32 v1, vcc, v2, v3

// SICI: v_subrev_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x4e,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_subrev_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x1b,0xd1,0x02,0x07,0x02,0x00]
v_subrev_i32 v1, s[0:1], v2, v3

// SICI: v_subrev_i32_e32 v1, vcc, v2, v3 ; encoding: [0x02,0x07,0x02,0x4e]
// VI:   v_subrev_i32_e32 v1, vcc, v2, v3 ; encoding: [0x02,0x07,0x02,0x36]
v_subrev_u32 v1, vcc, v2, v3

// SICI: v_subrev_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x4e,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_subrev_i32_e64 v1, s[0:1], v2, v3 ; encoding: [0x01,0x00,0x1b,0xd1,0x02,0x07,0x02,0x00]
v_subrev_u32 v1, s[0:1], v2, v3

// SICI: v_addc_u32_e32 v1, vcc, v2, v3, vcc ; encoding: [0x02,0x07,0x02,0x50]
// VI:   v_addc_u32_e32 v1, vcc, v2, v3, vcc ; encoding: [0x02,0x07,0x02,0x38]
v_addc_u32 v1, vcc, v2, v3, vcc

// SICI: v_addc_u32_e32 v1, vcc, v2, v3, vcc ; encoding: [0x02,0x07,0x02,0x50]
// VI:   v_addc_u32_e32 v1, vcc, v2, v3, vcc ; encoding: [0x02,0x07,0x02,0x38]
v_addc_u32_e32 v1, vcc, v2, v3, vcc


// SI: v_addc_u32_e64 v1, s[0:1], v2, v3, vcc ; encoding: [0x01,0x00,0x50,0xd2,0x02,0x07,0xaa,0x01]
// VI: v_addc_u32_e64 v1, s[0:1], v2, v3, vcc ; encoding: [0x01,0x00,0x1c,0xd1,0x02,0x07,0xaa,0x01]
v_addc_u32 v1, s[0:1], v2, v3, vcc

// SI: v_addc_u32_e64 v1, s[0:1], v2, v3, s[2:3] ; encoding: [0x01,0x00,0x50,0xd2,0x02,0x07,0x0a,0x00]
// VI: v_addc_u32_e64 v1, s[0:1], v2, v3, s[2:3] ; encoding: [0x01,0x00,0x1c,0xd1,0x02,0x07,0x0a,0x00]
v_addc_u32 v1, s[0:1], v2, v3, s[2:3]

// SI: 	v_addc_u32_e64 v1, s[0:1], v2, v3, s[2:3] ; encoding: [0x01,0x00,0x50,0xd2,0x02,0x07,0x0a,0x00]
// VI: v_addc_u32_e64 v1, s[0:1], v2, v3, s[2:3] ; encoding: [0x01,0x00,0x1c,0xd1,0x02,0x07,0x0a,0x00]
v_addc_u32_e64 v1, s[0:1], v2, v3, s[2:3]

// SI: v_addc_u32_e64 v1, s[0:1], v2, v3, -1 ; encoding: [0x01,0x00,0x50,0xd2,0x02,0x07,0x06,0x03]
// VI: v_addc_u32_e64 v1, s[0:1], v2, v3, -1 ; encoding: [0x01,0x00,0x1c,0xd1,0x02,0x07,0x06,0x03]
v_addc_u32_e64 v1, s[0:1], v2, v3, -1

// SI: v_addc_u32_e64 v1, vcc, v2, v3, -1 ; encoding: [0x01,0x6a,0x50,0xd2,0x02,0x07,0x06,0x03]
// VI: v_addc_u32_e64 v1, vcc, v2, v3, -1 ; encoding: [0x01,0x6a,0x1c,0xd1,0x02,0x07,0x06,0x03]
v_addc_u32_e64 v1, vcc, v2, v3, -1

// SI: v_addc_u32_e64 v1, vcc, v2, v3, vcc ; encoding: [0x01,0x6a,0x50,0xd2,0x02,0x07,0xaa,0x01]
// VI: v_addc_u32_e64 v1, vcc, v2, v3, vcc ; encoding: [0x01,0x6a,0x1c,0xd1,0x02,0x07,0xaa,0x01]
v_addc_u32_e64 v1, vcc, v2, v3, vcc

// SI: v_subb_u32_e32 v1, vcc, v2, v3, vcc ; encoding: [0x02,0x07,0x02,0x52]
// VI: v_subb_u32_e32 v1, vcc, v2, v3, vcc ; encoding: [0x02,0x07,0x02,0x3a]
v_subb_u32 v1, vcc, v2, v3, vcc

// SI: v_subb_u32_e64 v1, s[0:1], v2, v3, vcc ; encoding: [0x01,0x00,0x52,0xd2,0x02,0x07,0xaa,0x01]
// VI: v_subb_u32_e64 v1, s[0:1], v2, v3, vcc ; encoding: [0x01,0x00,0x1d,0xd1,0x02,0x07,0xaa,0x01]
v_subb_u32 v1, s[0:1], v2, v3, vcc

// SICI: v_subbrev_u32_e32 v1, vcc, v2, v3, vcc ; encoding: [0x02,0x07,0x02,0x54]
// VI:   v_subbrev_u32_e32 v1, vcc, v2, v3, vcc ; encoding: [0x02,0x07,0x02,0x3c]
v_subbrev_u32 v1, vcc, v2, v3, vcc

// SICI: v_subbrev_u32_e64 v1, s[0:1], v2, v3, vcc ; encoding: [0x01,0x00,0x54,0xd2,0x02,0x07,0xaa,0x01]
// VI: v_subbrev_u32_e64 v1, s[0:1], v2, v3, vcc ; encoding: [0x01,0x00,0x1e,0xd1,0x02,0x07,0xaa,0x01]
v_subbrev_u32 v1, s[0:1], v2, v3, vcc

// SICI: v_ldexp_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x56]
// VI:   v_ldexp_f32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x88,0xd2,0x02,0x07,0x02,0x00]
v_ldexp_f32 v1, v2, v3

// SICI: v_cvt_pkaccum_u8_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x58]
// VI:   v_cvt_pkaccum_u8_f32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0xf0,0xd1,0x02,0x07,0x02,0x00]
v_cvt_pkaccum_u8_f32 v1, v2, v3

// SICI: v_cvt_pknorm_i16_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x5a]
// VI:   v_cvt_pknorm_i16_f32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x94,0xd2,0x02,0x07,0x02,0x00]
v_cvt_pknorm_i16_f32 v1, v2, v3

// SICI: v_cvt_pknorm_u16_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x5c]
// VI:   v_cvt_pknorm_u16_f32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x95,0xd2,0x02,0x07,0x02,0x00]
v_cvt_pknorm_u16_f32 v1, v2, v3

// SICI: v_cvt_pkrtz_f16_f32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x5e]
// VI:   v_cvt_pkrtz_f16_f32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x96,0xd2,0x02,0x07,0x02,0x00]
v_cvt_pkrtz_f16_f32 v1, v2, v3

// SICI: v_cvt_pk_u16_u32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x60,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_cvt_pk_u16_u32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x97,0xd2,0x02,0x07,0x02,0x00]
v_cvt_pk_u16_u32_e64 v1, v2, v3

// SICI: v_cvt_pk_i16_i32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x62,0xd2,0x02,0x07,0x02,0x00]
// VI:   v_cvt_pk_i16_i32_e64 v1, v2, v3 ; encoding: [0x01,0x00,0x98,0xd2,0x02,0x07,0x02,0x00]
v_cvt_pk_i16_i32_e64 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_add_f16_e32 v1, v2, v3
// VI:     v_add_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x3e]
v_add_f16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_sub_f16_e32 v1, v2, v3
// VI:     v_sub_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x40]
v_sub_f16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_subrev_f16_e32 v1, v2, v3
// VI:     v_subrev_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x42]
v_subrev_f16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_mul_f16_e32 v1, v2, v3
// VI:     v_mul_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x44]
v_mul_f16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_mac_f16_e32 v1, v2, v3
// VI:     v_mac_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x46]
v_mac_f16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_madmk_f16 v1, v2, 64.0, v3
// VI:     v_madmk_f16_e32 v1, v2, 0x5400, v3 ; encoding: [0x02,0x07,0x02,0x48,0x00,0x54,0x00,0x00]
v_madmk_f16 v1, v2, 64.0, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_madak_f16 v1, v2, v3, 64.0
// VI:     v_madak_f16_e32 v1, v2, v3, 0x5400 ; encoding: [0x02,0x07,0x02,0x4a,0x00,0x54,0x00,0x00]
v_madak_f16 v1, v2, v3, 64.0

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_add_u16_e32 v1, v2, v3
// VI:     v_add_u16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x4c]
v_add_u16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_sub_u16_e32 v1, v2, v3
// VI:     v_sub_u16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x4e]
v_sub_u16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_subrev_u16_e32 v1, v2, v3
// VI:     v_subrev_u16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x50]
v_subrev_u16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_mul_lo_u16_e32 v1, v2, v3
// VI:     v_mul_lo_u16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x52]
v_mul_lo_u16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_lshlrev_b16_e32 v1, v2, v3
// VI:     v_lshlrev_b16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x54]
v_lshlrev_b16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_lshrrev_b16_e32 v1, v2, v3
// VI: v_lshrrev_b16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x56]
v_lshrrev_b16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_ashrrev_i16_e32 v1, v2, v3
// VI:     v_ashrrev_i16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x58]
v_ashrrev_i16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_max_f16_e32 v1, v2, v3
// VI:     v_max_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x5a]
v_max_f16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_min_f16_e32 v1, v2, v3
// VI:     v_min_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x5c]
v_min_f16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_max_u16_e32 v1, v2, v3
// VI:     v_max_u16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x5e]
v_max_u16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_max_i16_e32 v1, v2, v3
// VI:     v_max_i16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x60]
v_max_i16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_min_u16_e32 v1, v2, v3
// VI:     v_min_u16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x62]
v_min_u16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_min_i16_e32 v1, v2, v3
// VI:     v_min_i16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x64]
v_min_i16_e32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_ldexp_f16_e32 v1, v2, v3
// VI:     v_ldexp_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x66]
v_ldexp_f16_e32 v1, v2, v3
