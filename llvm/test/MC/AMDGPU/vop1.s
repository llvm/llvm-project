// RUN: not llvm-mc -triple=amdgcn -show-encoding %s | FileCheck %s --check-prefixes=GCN,SICI
// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck %s --check-prefixes=GCN,SICI
// RUN: not llvm-mc -triple=amdgcn -mcpu=bonaire -show-encoding %s | FileCheck %s --check-prefixes=GCN,CI,SICI,CIVI
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga -show-encoding %s | FileCheck %s --check-prefixes=GCN,CIVI,VI

// RUN: not llvm-mc -triple=amdgcn %s 2>&1 | FileCheck %s --check-prefixes=NOSI,NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck %s --check-prefixes=NOSI,NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck %s --check-prefix=NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga %s 2>&1 | FileCheck %s -check-prefix=NOVI --implicit-check-not=error:

// Force 32-bit encoding

// GCN: v_mov_b32_e32 v1, v2 ; encoding: [0x02,0x03,0x02,0x7e]
v_mov_b32_e32 v1, v2

// Force 32-bit encoding for special instructions
// FIXME: We should be printing _e32 suffixes for these:

// GCN: v_nop ; encoding: [0x00,0x00,0x00,0x7e]
v_nop_e32

// SICI: v_clrexcp ; encoding: [0x00,0x82,0x00,0x7e]
// VI:   v_clrexcp ; encoding: [0x00,0x6a,0x00,0x7e]
v_clrexcp_e32

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//


// GCN: v_nop ; encoding: [0x00,0x00,0x00,0x7e]
v_nop

// GCN: v_mov_b32_e32 v1, v2 ; encoding: [0x02,0x03,0x02,0x7e]
v_mov_b32_e32 v1, v2

// GCN: v_readfirstlane_b32 s1, v2 ; encoding: [0x02,0x05,0x02,0x7e]
v_readfirstlane_b32 s1, v2

// GCN: v_cvt_i32_f64_e32 v1, v[2:3] ; encoding: [0x02,0x07,0x02,0x7e]
v_cvt_i32_f64_e32 v1, v[2:3]

// GCN: v_cvt_f64_i32_e32 v[1:2], v2 ; encoding: [0x02,0x09,0x02,0x7e]
v_cvt_f64_i32_e32 v[1:2], v2

// GCN: v_cvt_f32_i32_e32 v1, v2 ; encoding: [0x02,0x0b,0x02,0x7e]
v_cvt_f32_i32_e32 v1, v2

// GCN: v_cvt_f32_u32_e32 v1, v2 ; encoding: [0x02,0x0d,0x02,0x7e]
v_cvt_f32_u32_e32 v1, v2

// GCN: v_cvt_u32_f32_e32 v1, v2 ; encoding: [0x02,0x0f,0x02,0x7e
v_cvt_u32_f32_e32 v1, v2

// GCN: v_cvt_i32_f32_e32 v1, v2 ; encoding: [0x02,0x11,0x02,0x7e]
v_cvt_i32_f32_e32 v1, v2

// GCN: v_cvt_f16_f32_e32 v1, v2 ; encoding: [0x02,0x15,0x02,0x7e]
v_cvt_f16_f32_e32 v1, v2

// GCN: v_cvt_f32_f16_e32 v1, v2 ; encoding: [0x02,0x17,0x02,0x7e]
v_cvt_f32_f16_e32 v1, v2

// GCN: v_cvt_rpi_i32_f32_e32 v1, v2 ; encoding: [0x02,0x19,0x02,0x7e]
v_cvt_rpi_i32_f32_e32 v1, v2

// GCN: v_cvt_flr_i32_f32_e32 v1, v2 ; encoding: [0x02,0x1b,0x02,0x7e]
v_cvt_flr_i32_f32_e32 v1, v2

// GCN: v_cvt_off_f32_i4_e32 v1, v2 ; encoding: [0x02,0x1d,0x02,0x7e]
v_cvt_off_f32_i4_e32 v1, v2

// GCN: v_cvt_f32_f64_e32 v1, v[2:3] ; encoding: [0x02,0x1f,0x02,0x7e]
v_cvt_f32_f64_e32 v1, v[2:3]

// GCN: v_cvt_f64_f32_e32 v[1:2], v2 ; encoding: [0x02,0x21,0x02,0x7e]
v_cvt_f64_f32_e32 v[1:2], v2

// GCN: v_cvt_f32_ubyte0_e32 v1, v2 ; encoding: [0x02,0x23,0x02,0x7e]
v_cvt_f32_ubyte0_e32 v1, v2

// GCN: v_cvt_f32_ubyte1_e32 v1, v2 ; encoding: [0x02,0x25,0x02,0x7e]
v_cvt_f32_ubyte1_e32 v1, v2

// GCN: v_cvt_f32_ubyte2_e32 v1, v2 ; encoding: [0x02,0x27,0x02,0x7e]
v_cvt_f32_ubyte2_e32 v1, v2

// GCN: v_cvt_f32_ubyte3_e32 v1, v2 ; encoding: [0x02,0x29,0x02,0x7e]
v_cvt_f32_ubyte3_e32 v1, v2

// GCN: v_cvt_u32_f64_e32 v1, v[2:3] ; encoding: [0x02,0x2b,0x02,0x7e]
v_cvt_u32_f64_e32 v1, v[2:3]

// GCN: v_cvt_f64_u32_e32 v[1:2], v2 ; encoding: [0x02,0x2d,0x02,0x7e]
v_cvt_f64_u32_e32 v[1:2], v2

// NOSI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSI: v_trunc_f64_e32 v[1:2], v[2:3]
// CIVI: v_trunc_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x2f,0x02,0x7e]
v_trunc_f64_e32 v[1:2], v[2:3]

// NOSI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSI: v_ceil_f64_e32 v[1:2], v[2:3]
// CIVI: v_ceil_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x31,0x02,0x7e]
v_ceil_f64_e32 v[1:2], v[2:3]

// NOSI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSI: v_rndne_f64_e32 v[1:2], v[2:3]
// CIVI: v_rndne_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x33,0x02,0x7e]
v_rndne_f64_e32 v[1:2], v[2:3]

// NOSI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSI: v_floor_f64_e32 v[1:2], v[2:3]
// CIVI: v_floor_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x35,0x02,0x7e]
v_floor_f64_e32 v[1:2], v[2:3]

// SICI: v_fract_f32_e32 v1, v2 ; encoding: [0x02,0x41,0x02,0x7e]
// VI:   v_fract_f32_e32 v1, v2 ; encoding: [0x02,0x37,0x02,0x7e]
v_fract_f32 v1, v2

// SICI: v_trunc_f32_e32 v1, v2 ; encoding: [0x02,0x43,0x02,0x7e]
// VI:   v_trunc_f32_e32 v1, v2 ; encoding: [0x02,0x39,0x02,0x7e]
v_trunc_f32 v1, v2

// SICI: v_ceil_f32_e32 v1, v2 ; encoding: [0x02,0x45,0x02,0x7e]
// VI:   v_ceil_f32_e32 v1, v2 ; encoding: [0x02,0x3b,0x02,0x7e]
v_ceil_f32 v1, v2

// SICI: v_rndne_f32_e32 v1, v2 ; encoding: [0x02,0x47,0x02,0x7e]
// VI:   v_rndne_f32_e32 v1, v2 ; encoding: [0x02,0x3d,0x02,0x7e]
v_rndne_f32 v1, v2

// SICI: v_floor_f32_e32 v1, v2 ; encoding: [0x02,0x49,0x02,0x7e]
// VI:   v_floor_f32_e32 v1, v2 ; encoding: [0x02,0x3f,0x02,0x7e]
v_floor_f32_e32 v1, v2

// SICI: v_exp_f32_e32 v1, v2 ; encoding: [0x02,0x4b,0x02,0x7e]
// VI:   v_exp_f32_e32 v1, v2 ; encoding: [0x02,0x41,0x02,0x7e]
v_exp_f32 v1, v2

// SICI: v_log_clamp_f32_e32 v1, v2 ; encoding: [0x02,0x4d,0x02,0x7e]
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: v_log_clamp_f32 v1, v2
v_log_clamp_f32 v1, v2

// SICI: v_log_f32_e32 v1, v2 ; encoding: [0x02,0x4f,0x02,0x7e]
// VI:   v_log_f32_e32 v1, v2 ; encoding: [0x02,0x43,0x02,0x7e]
v_log_f32 v1, v2

// SICI: v_rcp_clamp_f32_e32 v1, v2 ; encoding: [0x02,0x51,0x02,0x7e]
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: v_rcp_clamp_f32 v1, v2
v_rcp_clamp_f32 v1, v2

// SICI: v_rcp_legacy_f32_e32 v1, v2 ; encoding: [0x02,0x53,0x02,0x7e]
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: v_rcp_legacy_f32 v1, v2
v_rcp_legacy_f32 v1, v2

// SICI: v_rcp_f32_e32 v1, v2 ; encoding: [0x02,0x55,0x02,0x7e]
// VI:   v_rcp_f32_e32 v1, v2 ; encoding: [0x02,0x45,0x02,0x7e]
v_rcp_f32 v1, v2

// SICI: v_rcp_iflag_f32_e32 v1, v2 ; encoding: [0x02,0x57,0x02,0x7e]
// VI:   v_rcp_iflag_f32_e32 v1, v2 ; encoding: [0x02,0x47,0x02,0x7e]
v_rcp_iflag_f32 v1, v2

// SICI: v_rsq_clamp_f32_e32 v1, v2 ; encoding: [0x02,0x59,0x02,0x7e]
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: v_rsq_clamp_f32 v1, v2
v_rsq_clamp_f32 v1, v2

// SICI: v_rsq_legacy_f32_e32 v1, v2 ; encoding: [0x02,0x5b,0x02,0x7e]
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: v_rsq_legacy_f32 v1, v2
v_rsq_legacy_f32 v1, v2

// SICI: v_rsq_f32_e32 v1, v2 ; encoding: [0x02,0x5d,0x02,0x7e]
// VI:   v_rsq_f32_e32 v1, v2 ; encoding: [0x02,0x49,0x02,0x7e]
v_rsq_f32_e32 v1, v2

// SICI: v_rcp_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x5f,0x02,0x7e]
// VI:   v_rcp_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x4b,0x02,0x7e]
v_rcp_f64 v[1:2], v[2:3]

// SICI: v_rcp_clamp_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x61,0x02,0x7e]
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: v_rcp_clamp_f64 v[1:2], v[2:3]
v_rcp_clamp_f64 v[1:2], v[2:3]

// SICI: v_rsq_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x63,0x02,0x7e]
// VI:   v_rsq_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x4d,0x02,0x7e]
v_rsq_f64 v[1:2], v[2:3]

// SICI: v_rsq_clamp_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x65,0x02,0x7e]
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: v_rsq_clamp_f64 v[1:2], v[2:3]
v_rsq_clamp_f64 v[1:2], v[2:3]

// SICI: v_sqrt_f32_e32 v1, v2 ; encoding: [0x02,0x67,0x02,0x7e]
// VI:   v_sqrt_f32_e32 v1, v2 ; encoding: [0x02,0x4f,0x02,0x7e]
v_sqrt_f32 v1, v2

// SICI: v_sqrt_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x69,0x02,0x7e]
// VI:   v_sqrt_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x51,0x02,0x7e]
v_sqrt_f64 v[1:2], v[2:3]

// SICI: v_sin_f32_e32 v1, v2 ; encoding: [0x02,0x6b,0x02,0x7e]
// VI:   v_sin_f32_e32 v1, v2 ; encoding: [0x02,0x53,0x02,0x7e]
v_sin_f32 v1, v2

// SICI: v_cos_f32_e32 v1, v2 ; encoding: [0x02,0x6d,0x02,0x7e]
// VI:   v_cos_f32_e32 v1, v2 ; encoding: [0x02,0x55,0x02,0x7e]
v_cos_f32 v1, v2

// SICI: v_not_b32_e32 v1, v2 ; encoding: [0x02,0x6f,0x02,0x7e]
// VI:   v_not_b32_e32 v1, v2 ; encoding: [0x02,0x57,0x02,0x7e]
v_not_b32_e32 v1, v2

// SICI: v_bfrev_b32_e32 v1, v2 ; encoding: [0x02,0x71,0x02,0x7e]
// VI:   v_bfrev_b32_e32 v1, v2 ; encoding: [0x02,0x59,0x02,0x7e]
v_bfrev_b32_e32 v1, v2

// SICI: v_ffbh_u32_e32 v1, v2 ; encoding: [0x02,0x73,0x02,0x7e]
// VI:   v_ffbh_u32_e32 v1, v2 ; encoding: [0x02,0x5b,0x02,0x7e]
v_ffbh_u32_e32 v1, v2

// SICI: v_ffbl_b32_e32 v1, v2 ; encoding: [0x02,0x75,0x02,0x7e]
// VI:   v_ffbl_b32_e32 v1, v2 ; encoding: [0x02,0x5d,0x02,0x7e]
v_ffbl_b32_e32 v1, v2

// SICI: v_ffbh_i32_e32 v1, v2 ; encoding: [0x02,0x77,0x02,0x7e]
// VI:   v_ffbh_i32_e32 v1, v2 ; encoding: [0x02,0x5f,0x02,0x7e]
v_ffbh_i32_e32 v1, v2

// SICI: v_frexp_exp_i32_f64_e32 v1, v[2:3] ; encoding: [0x02,0x79,0x02,0x7e]
// VI:   v_frexp_exp_i32_f64_e32 v1, v[2:3] ; encoding: [0x02,0x61,0x02,0x7e]
v_frexp_exp_i32_f64_e32 v1, v[2:3]

// SICI: v_frexp_mant_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x7b,0x02,0x7e]
// VI:   v_frexp_mant_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x63,0x02,0x7e]
v_frexp_mant_f64_e32 v[1:2], v[2:3]

// SICI: v_fract_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x7d,0x02,0x7e]
// VI:   v_fract_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x65,0x02,0x7e]
v_fract_f64_e32 v[1:2], v[2:3]

// SICI: v_frexp_exp_i32_f32_e32 v1, v2 ; encoding: [0x02,0x7f,0x02,0x7e]
// VI:   v_frexp_exp_i32_f32_e32 v1, v2 ; encoding: [0x02,0x67,0x02,0x7e]
v_frexp_exp_i32_f32_e32 v1, v2

// SICI: v_frexp_mant_f32_e32 v1, v2 ; encoding: [0x02,0x81,0x02,0x7e]
// VI:   v_frexp_mant_f32_e32 v1, v2 ; encoding: [0x02,0x69,0x02,0x7e]
v_frexp_mant_f32 v1, v2

// SICI: v_clrexcp ; encoding: [0x00,0x82,0x00,0x7e]
// VI:   v_clrexcp ; encoding: [0x00,0x6a,0x00,0x7e]
v_clrexcp_e32

// SICI: v_movreld_b32_e32 v1, v2 ; encoding: [0x02,0x85,0x02,0x7e]
// VI:   v_movreld_b32_e32 v1, v2 ; encoding: [0x02,0x6d,0x02,0x7e]
v_movreld_b32_e32 v1, v2

// SICI: v_movrels_b32_e32 v1, v2 ; encoding: [0x02,0x87,0x02,0x7e]
// VI:   v_movrels_b32_e32 v1, v2 ; encoding: [0x02,0x6f,0x02,0x7e]
v_movrels_b32_e32 v1, v2

// SICI: v_movrelsd_b32_e32 v1, v2 ; encoding: [0x02,0x89,0x02,0x7e]
// VI:   v_movrelsd_b32_e32 v1, v2 ; encoding: [0x02,0x71,0x02,0x7e]
v_movrelsd_b32_e32 v1, v2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// NOVI: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_movrelsd_b32_e32 v1, s2

// NOSI: :[[@LINE+4]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSI: v_log_legacy_f32 v1, v2
// CI: v_log_legacy_f32_e32 v1, v2 ; encoding: [0x02,0x8b,0x02,0x7e]
// VI: v_log_legacy_f32_e32 v1, v2 ; encoding: [0x02,0x99,0x02,0x7e]
v_log_legacy_f32 v1, v2

// NOSI: :[[@LINE+4]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSI: v_exp_legacy_f32 v1, v2
// CI: v_exp_legacy_f32_e32 v1, v2 ; encoding: [0x02,0x8d,0x02,0x7e]
// VI: v_exp_legacy_f32_e32 v1, v2 ; encoding: [0x02,0x97,0x02,0x7e]
v_exp_legacy_f32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_cvt_f16_u16_e32 v1, v2
// VI: v_cvt_f16_u16_e32 v1, v2 ; encoding: [0x02,0x73,0x02,0x7e]
v_cvt_f16_u16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_cvt_f16_i16_e32 v1, v2
// VI: v_cvt_f16_i16_e32 v1, v2 ; encoding: [0x02,0x75,0x02,0x7e]
v_cvt_f16_i16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_cvt_u16_f16_e32 v1, v2
// VI: v_cvt_u16_f16_e32 v1, v2 ; encoding: [0x02,0x77,0x02,0x7e]
v_cvt_u16_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_cvt_i16_f16_e32 v1, v2
// VI: v_cvt_i16_f16_e32 v1, v2 ; encoding: [0x02,0x79,0x02,0x7e]
v_cvt_i16_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_rcp_f16_e32 v1, v2
// VI: v_rcp_f16_e32 v1, v2 ; encoding: [0x02,0x7b,0x02,0x7e]
v_rcp_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_sqrt_f16_e32 v1, v2
// VI: v_sqrt_f16_e32 v1, v2 ; encoding: [0x02,0x7d,0x02,0x7e]
v_sqrt_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_rsq_f16_e32 v1, v2
// VI: v_rsq_f16_e32 v1, v2 ; encoding: [0x02,0x7f,0x02,0x7e]
v_rsq_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_log_f16_e32 v1, v2
// VI: v_log_f16_e32 v1, v2 ; encoding: [0x02,0x81,0x02,0x7e]
v_log_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_exp_f16_e32 v1, v2
// VI: v_exp_f16_e32 v1, v2 ; encoding: [0x02,0x83,0x02,0x7e]
v_exp_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_frexp_mant_f16_e32 v1, v2
// VI: v_frexp_mant_f16_e32 v1, v2 ; encoding: [0x02,0x85,0x02,0x7e]
v_frexp_mant_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_frexp_exp_i16_f16_e32 v1, v2
// VI: v_frexp_exp_i16_f16_e32 v1, v2 ; encoding: [0x02,0x87,0x02,0x7e]
v_frexp_exp_i16_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_floor_f16_e32 v1, v2
// VI: v_floor_f16_e32 v1, v2 ; encoding: [0x02,0x89,0x02,0x7e]
v_floor_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_ceil_f16_e32 v1, v2
// VI: v_ceil_f16_e32 v1, v2 ; encoding: [0x02,0x8b,0x02,0x7e]
v_ceil_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_trunc_f16_e32 v1, v2
// VI: v_trunc_f16_e32 v1, v2 ; encoding: [0x02,0x8d,0x02,0x7e]
v_trunc_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_rndne_f16_e32 v1, v2
// VI: v_rndne_f16_e32 v1, v2 ; encoding: [0x02,0x8f,0x02,0x7e]
v_rndne_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_fract_f16_e32 v1, v2
// VI: v_fract_f16_e32 v1, v2 ; encoding: [0x02,0x91,0x02,0x7e]
v_fract_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_sin_f16_e32 v1, v2
// VI: v_sin_f16_e32 v1, v2 ; encoding: [0x02,0x93,0x02,0x7e]
v_sin_f16_e32 v1, v2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOSICI: v_cos_f16_e32 v1, v2
// VI: v_cos_f16_e32 v1, v2 ; encoding: [0x02,0x95,0x02,0x7e]
v_cos_f16_e32 v1, v2
