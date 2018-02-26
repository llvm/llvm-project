// RUN: not llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=CIVI --check-prefix=VI

// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s 2>&1 | FileCheck %s -check-prefix=NOVI

v_mov_b32 [v1], [v2]
// GCN:  v_mov_b32_e32 v1, v2 ; encoding: [0x02,0x03,0x02,0x7e]

v_mov_b32 v0, 0.5
// GCN: v_mov_b32_e32 v0, 0.5 ; encoding: [0xf0,0x02,0x00,0x7e]

v_mov_b32_e32 v0, 3.125
// GCN: v_mov_b32_e32 v0, 0x40480000 ; encoding: [0xff,0x02,0x00,0x7e,0x00,0x00,0x48,0x40]

v_mov_b32     v1, ttmp8
// SICI: v_mov_b32_e32 v1, ttmp8         ; encoding: [0x78,0x02,0x02,0x7e]
// VI:   v_mov_b32_e32 v1, ttmp8         ; encoding: [0x78,0x02,0x02,0x7e]

// GCN: v_mov_b32_e32 v1, v2 ; encoding: [0x02,0x03,0x02,0x7e]
v_mov_b32 v1, v2

// SICI: v_not_b32_e32 v1, v2 ; encoding: [0x02,0x6f,0x02,0x7e]
// VI:   v_not_b32_e32 v1, v2 ; encoding: [0x02,0x57,0x02,0x7e]
v_not_b32 v1, v2

// SICI: v_bfrev_b32_e32 v1, v2 ; encoding: [0x02,0x71,0x02,0x7e]
// VI:   v_bfrev_b32_e32 v1, v2 ; encoding: [0x02,0x59,0x02,0x7e]
v_bfrev_b32 v1, v2

// SICI: v_ffbh_u32_e32 v1, v2 ; encoding: [0x02,0x73,0x02,0x7e]
// VI:   v_ffbh_u32_e32 v1, v2 ; encoding: [0x02,0x5b,0x02,0x7e]
v_ffbh_u32 v1, v2

// SICI: v_ffbl_b32_e32 v1, v2 ; encoding: [0x02,0x75,0x02,0x7e]
// VI:   v_ffbl_b32_e32 v1, v2 ; encoding: [0x02,0x5d,0x02,0x7e]
v_ffbl_b32 v1, v2

// SICI: v_ffbh_i32_e32 v1, v2 ; encoding: [0x02,0x77,0x02,0x7e]
// VI:   v_ffbh_i32_e32 v1, v2 ; encoding: [0x02,0x5f,0x02,0x7e]
v_ffbh_i32_e32 v1, v2

// SICI: v_frexp_exp_i32_f64_e32 v1, v[2:3] ; encoding: [0x02,0x79,0x02,0x7e]
// VI:   v_frexp_exp_i32_f64_e32 v1, v[2:3] ; encoding: [0x02,0x61,0x02,0x7e]
v_frexp_exp_i32_f64 v1, v[2:3]

// SICI: v_frexp_mant_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x7b,0x02,0x7e]
// VI;   v_frexp_mant_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x63,0x02,0x7e]
v_frexp_mant_f64 v[1:2], v[2:3]

// SICI: v_fract_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x7d,0x02,0x7e]
// VI:   v_fract_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x65,0x02,0x7e]
v_fract_f64 v[1:2], v[2:3]

// SICI: v_frexp_exp_i32_f32_e32 v1, v2 ; encoding: [0x02,0x7f,0x02,0x7e]
// VI:   v_frexp_exp_i32_f32_e32 v1, v2 ; encoding: [0x02,0x67,0x02,0x7e]
v_frexp_exp_i32_f32 v1, v2

// SICI: v_frexp_mant_f32_e32 v1, v2 ; encoding: [0x02,0x81,0x02,0x7e]
// VI:   v_frexp_mant_f32_e32 v1, v2 ; encoding: [0x02,0x69,0x02,0x7e]
v_frexp_mant_f32 v1, v2

// SICI: v_clrexcp ; encoding: [0x00,0x82,0x00,0x7e]
// VI:   v_clrexcp ; encoding: [0x00,0x6a,0x00,0x7e]
v_clrexcp

// SICI: v_movreld_b32_e32 v1, v2 ; encoding: [0x02,0x85,0x02,0x7e]
// VI:   v_movreld_b32_e32 v1, v2 ; encoding: [0x02,0x6d,0x02,0x7e]
v_movreld_b32 v1, v2

// SICI: v_movrels_b32_e32 v1, v2 ; encoding: [0x02,0x87,0x02,0x7e]
// VI:   v_movrels_b32_e32 v1, v2 ; encoding: [0x02,0x6f,0x02,0x7e]
v_movrels_b32 v1, v2

// SICI: v_movrelsd_b32_e32 v1, v2 ; encoding: [0x02,0x89,0x02,0x7e]
// VI:   v_movrelsd_b32_e32 v1, v2 ; encoding: [0x02,0x71,0x02,0x7e]
v_movrelsd_b32 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_cvt_f16_u16 v1, v2
// VI: v_cvt_f16_u16_e32 v1, v2 ; encoding: [0x02,0x73,0x02,0x7e]
v_cvt_f16_u16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_cvt_f16_i16 v1, v2
// VI: v_cvt_f16_i16_e32 v1, v2 ; encoding: [0x02,0x75,0x02,0x7e]
v_cvt_f16_i16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_cvt_u16_f16 v1, v2
// VI: v_cvt_u16_f16_e32 v1, v2 ; encoding: [0x02,0x77,0x02,0x7e]
v_cvt_u16_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_cvt_i16_f16 v1, v2
// VI: v_cvt_i16_f16_e32 v1, v2 ; encoding: [0x02,0x79,0x02,0x7e]
v_cvt_i16_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_rcp_f16 v1, v2
// VI: v_rcp_f16_e32 v1, v2 ; encoding: [0x02,0x7b,0x02,0x7e]
v_rcp_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_sqrt_f16 v1, v2
// VI: v_sqrt_f16_e32 v1, v2 ; encoding: [0x02,0x7d,0x02,0x7e]
v_sqrt_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_rsq_f16 v1, v2
// VI: v_rsq_f16_e32 v1, v2 ; encoding: [0x02,0x7f,0x02,0x7e]
v_rsq_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_log_f16 v1, v2
// VI: v_log_f16_e32 v1, v2 ; encoding: [0x02,0x81,0x02,0x7e]
v_log_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_exp_f16 v1, v2
// VI: v_exp_f16_e32 v1, v2 ; encoding: [0x02,0x83,0x02,0x7e]
v_exp_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_frexp_mant_f16 v1, v2
// VI: v_frexp_mant_f16_e32 v1, v2 ; encoding: [0x02,0x85,0x02,0x7e]
v_frexp_mant_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_frexp_exp_i16_f16 v1, v2
// VI: v_frexp_exp_i16_f16_e32 v1, v2 ; encoding: [0x02,0x87,0x02,0x7e]
v_frexp_exp_i16_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_floor_f16 v1, v2
// VI: v_floor_f16_e32 v1, v2 ; encoding: [0x02,0x89,0x02,0x7e]
v_floor_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_ceil_f16 v1, v2
// VI: v_ceil_f16_e32 v1, v2 ; encoding: [0x02,0x8b,0x02,0x7e]
v_ceil_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_trunc_f16 v1, v2
// VI: v_trunc_f16_e32 v1, v2 ; encoding: [0x02,0x8d,0x02,0x7e]
v_trunc_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_rndne_f16 v1, v2
// VI: v_rndne_f16_e32 v1, v2 ; encoding: [0x02,0x8f,0x02,0x7e]
v_rndne_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_fract_f16 v1, v2
// VI: v_fract_f16_e32 v1, v2 ; encoding: [0x02,0x91,0x02,0x7e]
v_fract_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_sin_f16 v1, v2
// VI: v_sin_f16_e32 v1, v2 ; encoding: [0x02,0x93,0x02,0x7e]
v_sin_f16 v1, v2

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_cos_f16 v1, v2
// VI: v_cos_f16_e32 v1, v2 ; encoding: [0x02,0x95,0x02,0x7e]
v_cos_f16 v1, v2

// src0 inline
// SICI: v_mul_i32_i24_e32 v1, 3, v3 ; encoding: [0x83,0x06,0x02,0x12]
v_mul_i32_i24 v1, 3, v3

// src0 negative inline
// SICI: v_mul_i32_i24_e32 v1, -3, v3 ; encoding: [0xc3,0x06,0x02,0x12]
v_mul_i32_i24 v1, -3, v3

// src1 inline
// SICI: v_mul_i32_i24_e64 v1, v2, 3 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x07,0x01,0x00]
v_mul_i32_i24 v1, v2, 3

// src1 negative inline
// SICI: v_mul_i32_i24_e64 v1, v2, -3 ; encoding: [0x01,0x00,0x12,0xd2,0x02,0x87,0x01,0x00]
v_mul_i32_i24 v1, v2, -3

// GCN: v_cvt_flr_i32_f32_e32 v1, v2 ; encoding: [0x02,0x1b,0x02,0x7e]
v_cvt_flr_i32_f32 v1, v2

// GCN: v_cvt_off_f32_i4_e32 v1, v2 ; encoding: [0x02,0x1d,0x02,0x7e]
v_cvt_off_f32_i4_e32 v1, v2

// GCN: v_cvt_f32_f64_e32 v1, v[2:3] ; encoding: [0x02,0x1f,0x02,0x7e]
v_cvt_f32_f64 v1, v[2:3]

// GCN: v_cvt_f64_f32_e32 v[1:2], v2 ; encoding: [0x02,0x21,0x02,0x7e]
v_cvt_f64_f32 v[1:2], v2

// GCN: v_cvt_f32_ubyte0_e32 v1, v2 ; encoding: [0x02,0x23,0x02,0x7e]
v_cvt_f32_ubyte0 v1, v2

// GCN: v_cvt_f32_ubyte1_e32 v1, v2 ; encoding: [0x02,0x25,0x02,0x7e]
v_cvt_f32_ubyte1_e32 v1, v2

// GCN: v_cvt_f32_ubyte2_e32 v1, v2 ; encoding: [0x02,0x27,0x02,0x7e]
v_cvt_f32_ubyte2 v1, v2

// GCN: v_cvt_f32_ubyte3_e32 v1, v2 ; encoding: [0x02,0x29,0x02,0x7e]
v_cvt_f32_ubyte3 v1, v2

// GCN: v_cvt_u32_f64_e32 v1, v[2:3] ; encoding: [0x02,0x2b,0x02,0x7e]
v_cvt_u32_f64 v1, v[2:3]

// GCN: v_cvt_f64_u32_e32 v[1:2], v2 ; encoding: [0x02,0x2d,0x02,0x7e]
v_cvt_f64_u32 v[1:2], v2

// SICI: v_mul_i32_i24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x12]
// VI:   v_mul_i32_i24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x0c]
v_mul_i32_i24 v1, v2, v3

// SICI: v_mul_hi_i32_i24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x14]
// VI:   v_mul_hi_i32_i24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x0e]
v_mul_hi_i32_i24 v1, v2, v3

// SICI: v_mul_u32_u24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x16]
// VI:   v_mul_u32_u24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x10]
v_mul_u32_u24 v1, v2, v3

// SICI: v_mul_hi_u32_u24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x18]
// VI:   v_mul_hi_u32_u24_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x12]
v_mul_hi_u32_u24 v1, v2, v3

// SICI: v_min_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x22]
// VI:   v_min_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x18]
v_min_i32 v1, v2, v3

// SICI: v_max_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x24]
// VI:   v_max_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x1a]
v_max_i32 v1, v2, v3

// SICI: v_min_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x26]
// VI:   v_min_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x1c]
v_min_u32 v1, v2, v3

// SICI: v_max_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x28]
// VI:   v_max_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x1e]
v_max_u32 v1, v2, v3

// SICI: v_lshr_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x2a]
// NOVI: error: instruction not supported on this GPU
// NOVI: v_lshr_b32 v1, v2, v3
v_lshr_b32 v1, v2, v3

// SICI: v_lshrrev_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x2c]
// VI:   v_lshrrev_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x20]
v_lshrrev_b32 v1, v2, v3

// SICI: v_ashr_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x2e]
// NOVI: error: instruction not supported on this GPU
// NOVI: v_ashr_i32 v1, v2, v3
v_ashr_i32 v1, v2, v3

// SICI: v_ashrrev_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x30]
// VI:   v_ashrrev_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x22]
v_ashrrev_i32 v1, v2, v3

// SICI: v_lshl_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x32]
// NOVI: error: instruction not supported on this GPU
// NOVI: v_lshl_b32_e32 v1, v2, v3
v_lshl_b32_e32 v1, v2, v3

// SICI: v_lshlrev_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x34]
// VI:   v_lshlrev_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x24]
v_lshlrev_b32 v1, v2, v3

// SICI: v_and_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x36]
// VI:   v_and_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x26]
v_and_b32 v1, v2, v3

// SICI: v_or_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x38]
// VI:   v_or_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x28]
v_or_b32 v1, v2, v3

// SICI: v_xor_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x3a]
// VI:   v_xor_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x2a]
v_xor_b32 v1, v2, v3

// SICI: v_bfm_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x3c]
// VI:   v_bfm_b32 v1, v2, v3 ; encoding: [0x01,0x00,0x93,0xd2,0x02,0x07,0x02,0x00]
v_bfm_b32 v1, v2, v3

// SICI: v_bcnt_u32_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x44]
// VI:   v_bcnt_u32_b32 v1, v2, v3 ; encoding: [0x01,0x00,0x8b,0xd2,0x02,0x07,0x02,0x00]
v_bcnt_u32_b32 v1, v2, v3

// SICI: v_mbcnt_lo_u32_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x46]
// VI:   v_mbcnt_lo_u32_b32 v1, v2, v3 ; encoding: [0x01,0x00,0x8c,0xd2,0x02,0x07,0x02,0x00]
v_mbcnt_lo_u32_b32 v1, v2, v3

// SICI: v_mbcnt_hi_u32_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x48]
// VI:   v_mbcnt_hi_u32_b32 v1, v2, v3 ; encoding: [0x01,0x00,0x8d,0xd2,0x02,0x07,0x02,0x00]
v_mbcnt_hi_u32_b32 v1, v2, v3

// SICI: v_cvt_pk_u16_u32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x60]
// VI:   v_cvt_pk_u16_u32 v1, v2, v3 ; encoding: [0x01,0x00,0x97,0xd2,0x02,0x07,0x02,0x00]
v_cvt_pk_u16_u32 v1, v2, v3

// SICI: v_cvt_pk_i16_i32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x62]
// VI:   v_cvt_pk_i16_i32 v1, v2, v3 ; encoding: [0x01,0x00,0x98,0xd2,0x02,0x07,0x02,0x00]
v_cvt_pk_i16_i32 v1, v2, v3

// SICI: v_bfm_b32_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x3c]
// VI:   v_bfm_b32 v1, v2, v3 ; encoding: [0x01,0x00,0x93,0xd2,0x02,0x07,0x02,0x00]
v_bfm_b32 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_add_f16 v1, v2, v3
// VI:     v_add_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x3e]
v_add_f16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_sub_f16 v1, v2, v3
// VI:     v_sub_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x40]
v_sub_f16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_subrev_f16 v1, v2, v3
// VI:     v_subrev_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x42]
v_subrev_f16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_mul_f16 v1, v2, v3
// VI:     v_mul_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x44]
v_mul_f16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_mac_f16 v1, v2, v3
// VI:     v_mac_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x46]
v_mac_f16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_add_u16 v1, v2, v3
// VI:     v_add_u16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x4c]
v_add_u16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_sub_u16 v1, v2, v3
// VI:     v_sub_u16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x4e]
v_sub_u16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_subrev_u16 v1, v2, v3
// VI:     v_subrev_u16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x50]
v_subrev_u16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_mul_lo_u16 v1, v2, v3
// VI:     v_mul_lo_u16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x52]
v_mul_lo_u16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_lshlrev_b16 v1, v2, v3
// VI:     v_lshlrev_b16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x54]
v_lshlrev_b16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_lshrrev_b16 v1, v2, v3
// VI: v_lshrrev_b16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x56]
v_lshrrev_b16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_ashrrev_i16 v1, v2, v3
// VI:     v_ashrrev_i16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x58]
v_ashrrev_i16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_max_f16 v1, v2, v3
// VI:     v_max_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x5a]
v_max_f16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_min_f16 v1, v2, v3
// VI:     v_min_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x5c]
v_min_f16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_max_u16 v1, v2, v3
// VI:     v_max_u16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x5e]
v_max_u16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_max_i16 v1, v2, v3
// VI:     v_max_i16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x60]
v_max_i16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_min_u16 v1, v2, v3
// VI:     v_min_u16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x62]
v_min_u16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_min_i16 v1, v2, v3
// VI:     v_min_i16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x64]
v_min_i16 v1, v2, v3

// NOSICI: error: instruction not supported on this GPU
// NOSICI: v_ldexp_f16 v1, v2, v3
// VI:     v_ldexp_f16_e32 v1, v2, v3 ; encoding: [0x02,0x07,0x02,0x66]
v_ldexp_f16 v1, v2, v3
