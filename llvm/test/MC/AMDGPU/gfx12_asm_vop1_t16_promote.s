// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1200 -mattr=+real-true16 -show-encoding %s | FileCheck --check-prefix=GFX12 --implicit-check-not=_e32 %s

v_ceil_f16 v128, 0xfe0b
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, -1
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, 0.5
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, exec_hi
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, exec_lo
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, m0
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, null
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, s1
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, s105
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, src_scc
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, ttmp15
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, v1
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, v127
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, vcc_hi
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, vcc_lo
// GFX12: v_ceil_f16_e64

v_ceil_f16 v5, v199
// GFX12: v_ceil_f16_e64

v_cos_f16 v128, 0xfe0b
// GFX12: v_cos_f16_e64

v_cos_f16 v255, -1
// GFX12: v_cos_f16_e64

v_cos_f16 v255, 0.5
// GFX12: v_cos_f16_e64

v_cos_f16 v255, exec_hi
// GFX12: v_cos_f16_e64

v_cos_f16 v255, exec_lo
// GFX12: v_cos_f16_e64

v_cos_f16 v255, m0
// GFX12: v_cos_f16_e64

v_cos_f16 v255, null
// GFX12: v_cos_f16_e64

v_cos_f16 v255, s1
// GFX12: v_cos_f16_e64

v_cos_f16 v255, s105
// GFX12: v_cos_f16_e64

v_cos_f16 v255, src_scc
// GFX12: v_cos_f16_e64

v_cos_f16 v255, ttmp15
// GFX12: v_cos_f16_e64

v_cos_f16 v255, v1
// GFX12: v_cos_f16_e64

v_cos_f16 v255, v127
// GFX12: v_cos_f16_e64

v_cos_f16 v255, vcc_hi
// GFX12: v_cos_f16_e64

v_cos_f16 v255, vcc_lo
// GFX12: v_cos_f16_e64

v_cos_f16 v5, v199
// GFX12: v_cos_f16_e64

v_cvt_f16_f32 v255.l, v1
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.l, v255
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v128.h, 0xaf123456
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, -1
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, 0.5
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, exec_hi
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, exec_lo
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, m0
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, null
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, s1
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, s105
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, src_scc
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, ttmp15
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, v1
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, v255
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, vcc_hi
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, vcc_lo
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_i16 v128, 0xfe0b
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, -1
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, 0.5
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, exec_hi
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, exec_lo
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, m0
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, null
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, s1
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, s105
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, src_scc
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, ttmp15
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, v1
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, v127
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, vcc_hi
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, vcc_lo
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v5, v199
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_u16 v128, 0xfe0b
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, -1
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, 0.5
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, exec_hi
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, exec_lo
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, m0
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, null
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, s1
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, s105
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, src_scc
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, ttmp15
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, v1
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, v127
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, vcc_hi
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, vcc_lo
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v5, v199
// GFX12: v_cvt_f16_u16_e64

v_cvt_f32_f16 v5, v199.l
// GFX12: v_cvt_f32_f16_e64

v_cvt_f32_f16 v5, v199.h
// GFX12: v_cvt_f32_f16_e64

v_cvt_i16_f16 v128, 0xfe0b
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, -1
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, 0.5
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, exec_hi
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, exec_lo
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, m0
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, null
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, s1
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, s105
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, src_scc
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, ttmp15
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, v1
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, v127
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, vcc_hi
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, vcc_lo
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v5, v199
// GFX12: v_cvt_i16_f16_e64

v_cvt_i32_i16 v5, v199
// GFX12: v_cvt_i32_i16_e64

v_cvt_norm_i16_f16 v128, 0xfe0b
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, -1
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, 0.5
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, exec_hi
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, exec_lo
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, m0
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, null
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, s1
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, s105
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, src_scc
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, ttmp15
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, v1
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, v127
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, vcc_hi
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, vcc_lo
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v5, v199
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_u16_f16 v128, 0xfe0b
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, -1
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, 0.5
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, exec_hi
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, exec_lo
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, m0
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, null
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, s1
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, s105
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, src_scc
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, ttmp15
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, v1
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, v127
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, vcc_hi
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, vcc_lo
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v5, v199
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_u16_f16 v128, 0xfe0b
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, -1
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, 0.5
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, exec_hi
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, exec_lo
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, m0
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, null
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, s1
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, s105
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, src_scc
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, ttmp15
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, v1
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, v127
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, vcc_hi
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, vcc_lo
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v5, v199
// GFX12: v_cvt_u16_f16_e64

v_cvt_u32_u16 v5, v199
// GFX12: v_cvt_u32_u16_e64

v_exp_f16 v128, 0xfe0b
// GFX12: v_exp_f16_e64

v_exp_f16 v255, -1
// GFX12: v_exp_f16_e64

v_exp_f16 v255, 0.5
// GFX12: v_exp_f16_e64

v_exp_f16 v255, exec_hi
// GFX12: v_exp_f16_e64

v_exp_f16 v255, exec_lo
// GFX12: v_exp_f16_e64

v_exp_f16 v255, m0
// GFX12: v_exp_f16_e64

v_exp_f16 v255, null
// GFX12: v_exp_f16_e64

v_exp_f16 v255, s1
// GFX12: v_exp_f16_e64

v_exp_f16 v255, s105
// GFX12: v_exp_f16_e64

v_exp_f16 v255, src_scc
// GFX12: v_exp_f16_e64

v_exp_f16 v255, ttmp15
// GFX12: v_exp_f16_e64

v_exp_f16 v255, v1
// GFX12: v_exp_f16_e64

v_exp_f16 v255, v127
// GFX12: v_exp_f16_e64

v_exp_f16 v255, vcc_hi
// GFX12: v_exp_f16_e64

v_exp_f16 v255, vcc_lo
// GFX12: v_exp_f16_e64

v_exp_f16 v5, v199
// GFX12: v_exp_f16_e64

v_floor_f16 v128, 0xfe0b
// GFX12: v_floor_f16_e64

v_floor_f16 v255, -1
// GFX12: v_floor_f16_e64

v_floor_f16 v255, 0.5
// GFX12: v_floor_f16_e64

v_floor_f16 v255, exec_hi
// GFX12: v_floor_f16_e64

v_floor_f16 v255, exec_lo
// GFX12: v_floor_f16_e64

v_floor_f16 v255, m0
// GFX12: v_floor_f16_e64

v_floor_f16 v255, null
// GFX12: v_floor_f16_e64

v_floor_f16 v255, s1
// GFX12: v_floor_f16_e64

v_floor_f16 v255, s105
// GFX12: v_floor_f16_e64

v_floor_f16 v255, src_scc
// GFX12: v_floor_f16_e64

v_floor_f16 v255, ttmp15
// GFX12: v_floor_f16_e64

v_floor_f16 v255, v1
// GFX12: v_floor_f16_e64

v_floor_f16 v255, v127
// GFX12: v_floor_f16_e64

v_floor_f16 v255, vcc_hi
// GFX12: v_floor_f16_e64

v_floor_f16 v255, vcc_lo
// GFX12: v_floor_f16_e64

v_floor_f16 v5, v199
// GFX12: v_floor_f16_e64

v_fract_f16 v128, 0xfe0b
// GFX12: v_fract_f16_e64

v_fract_f16 v255, -1
// GFX12: v_fract_f16_e64

v_fract_f16 v255, 0.5
// GFX12: v_fract_f16_e64

v_fract_f16 v255, exec_hi
// GFX12: v_fract_f16_e64

v_fract_f16 v255, exec_lo
// GFX12: v_fract_f16_e64

v_fract_f16 v255, m0
// GFX12: v_fract_f16_e64

v_fract_f16 v255, null
// GFX12: v_fract_f16_e64

v_fract_f16 v255, s1
// GFX12: v_fract_f16_e64

v_fract_f16 v255, s105
// GFX12: v_fract_f16_e64

v_fract_f16 v255, src_scc
// GFX12: v_fract_f16_e64

v_fract_f16 v255, ttmp15
// GFX12: v_fract_f16_e64

v_fract_f16 v255, v1
// GFX12: v_fract_f16_e64

v_fract_f16 v255, v127
// GFX12: v_fract_f16_e64

v_fract_f16 v255, vcc_hi
// GFX12: v_fract_f16_e64

v_fract_f16 v255, vcc_lo
// GFX12: v_fract_f16_e64

v_fract_f16 v5, v199
// GFX12: v_fract_f16_e64

v_frexp_exp_i16_f16 v128, 0xfe0b
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, -1
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, 0.5
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, exec_hi
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, exec_lo
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, m0
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, null
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, s1
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, s105
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, src_scc
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, ttmp15
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, v1
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, v127
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, vcc_hi
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, vcc_lo
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v5, v199
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_mant_f16 v128, 0xfe0b
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, -1
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, 0.5
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, exec_hi
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, exec_lo
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, m0
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, null
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, s1
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, s105
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, src_scc
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, ttmp15
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, v1
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, v127
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, vcc_hi
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, vcc_lo
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v5, v199
// GFX12: v_frexp_mant_f16_e64

v_log_f16 v128, 0xfe0b
// GFX12: v_log_f16_e64

v_log_f16 v255, -1
// GFX12: v_log_f16_e64

v_log_f16 v255, 0.5
// GFX12: v_log_f16_e64

v_log_f16 v255, exec_hi
// GFX12: v_log_f16_e64

v_log_f16 v255, exec_lo
// GFX12: v_log_f16_e64

v_log_f16 v255, m0
// GFX12: v_log_f16_e64

v_log_f16 v255, null
// GFX12: v_log_f16_e64

v_log_f16 v255, s1
// GFX12: v_log_f16_e64

v_log_f16 v255, s105
// GFX12: v_log_f16_e64

v_log_f16 v255, src_scc
// GFX12: v_log_f16_e64

v_log_f16 v255, ttmp15
// GFX12: v_log_f16_e64

v_log_f16 v255, v1
// GFX12: v_log_f16_e64

v_log_f16 v255, v127
// GFX12: v_log_f16_e64

v_log_f16 v255, vcc_hi
// GFX12: v_log_f16_e64

v_log_f16 v255, vcc_lo
// GFX12: v_log_f16_e64

v_log_f16 v5, v199
// GFX12: v_log_f16_e64

v_not_b16 v128, 0xfe0b
// GFX12: v_not_b16_e64

v_not_b16 v255, -1
// GFX12: v_not_b16_e64

v_not_b16 v255, 0.5
// GFX12: v_not_b16_e64

v_not_b16 v255, exec_hi
// GFX12: v_not_b16_e64

v_not_b16 v255, exec_lo
// GFX12: v_not_b16_e64

v_not_b16 v255, m0
// GFX12: v_not_b16_e64

v_not_b16 v255, null
// GFX12: v_not_b16_e64

v_not_b16 v255, s1
// GFX12: v_not_b16_e64

v_not_b16 v255, s105
// GFX12: v_not_b16_e64

v_not_b16 v255, src_scc
// GFX12: v_not_b16_e64

v_not_b16 v255, ttmp15
// GFX12: v_not_b16_e64

v_not_b16 v255, v1
// GFX12: v_not_b16_e64

v_not_b16 v255, v127
// GFX12: v_not_b16_e64

v_not_b16 v255, vcc_hi
// GFX12: v_not_b16_e64

v_not_b16 v255, vcc_lo
// GFX12: v_not_b16_e64

v_not_b16 v5, v199
// GFX12: v_not_b16_e64

v_rcp_f16 v128, 0xfe0b
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, -1
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, 0.5
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, exec_hi
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, exec_lo
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, m0
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, null
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, s1
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, s105
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, src_scc
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, ttmp15
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, v1
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, v127
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, vcc_hi
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, vcc_lo
// GFX12: v_rcp_f16_e64

v_rcp_f16 v5, v199
// GFX12: v_rcp_f16_e64

v_rndne_f16 v128, 0xfe0b
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, -1
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, 0.5
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, exec_hi
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, exec_lo
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, m0
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, null
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, s1
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, s105
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, src_scc
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, ttmp15
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, v1
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, v127
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, vcc_hi
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, vcc_lo
// GFX12: v_rndne_f16_e64

v_rndne_f16 v5, v199
// GFX12: v_rndne_f16_e64

v_rsq_f16 v128, 0xfe0b
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, -1
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, 0.5
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, exec_hi
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, exec_lo
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, m0
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, null
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, s1
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, s105
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, src_scc
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, ttmp15
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, v1
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, v127
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, vcc_hi
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, vcc_lo
// GFX12: v_rsq_f16_e64

v_rsq_f16 v5, v199
// GFX12: v_rsq_f16_e64

v_sat_pk_u8_i16 v199, v5
// GFX12: v_sat_pk_u8_i16_e64

v_sin_f16 v128, 0xfe0b
// GFX12: v_sin_f16_e64

v_sin_f16 v255, -1
// GFX12: v_sin_f16_e64

v_sin_f16 v255, 0.5
// GFX12: v_sin_f16_e64

v_sin_f16 v255, exec_hi
// GFX12: v_sin_f16_e64

v_sin_f16 v255, exec_lo
// GFX12: v_sin_f16_e64

v_sin_f16 v255, m0
// GFX12: v_sin_f16_e64

v_sin_f16 v255, null
// GFX12: v_sin_f16_e64

v_sin_f16 v255, s1
// GFX12: v_sin_f16_e64

v_sin_f16 v255, s105
// GFX12: v_sin_f16_e64

v_sin_f16 v255, src_scc
// GFX12: v_sin_f16_e64

v_sin_f16 v255, ttmp15
// GFX12: v_sin_f16_e64

v_sin_f16 v255, v1
// GFX12: v_sin_f16_e64

v_sin_f16 v255, v127
// GFX12: v_sin_f16_e64

v_sin_f16 v255, vcc_hi
// GFX12: v_sin_f16_e64

v_sin_f16 v255, vcc_lo
// GFX12: v_sin_f16_e64

v_sin_f16 v5, v199
// GFX12: v_sin_f16_e64

v_sqrt_f16 v128, 0xfe0b
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, -1
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, 0.5
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, exec_hi
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, exec_lo
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, m0
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, null
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, s1
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, s105
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, src_scc
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, ttmp15
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, v1
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, v127
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, vcc_hi
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, vcc_lo
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v5, v199
// GFX12: v_sqrt_f16_e64

v_trunc_f16 v128, 0xfe0b
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, -1
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, 0.5
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, exec_hi
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, exec_lo
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, m0
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, null
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, s1
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, s105
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, src_scc
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, ttmp15
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, v1
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, v127
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, vcc_hi
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, vcc_lo
// GFX12: v_trunc_f16_e64

v_trunc_f16 v5, v199
// GFX12: v_trunc_f16_e64

v_ceil_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_ceil_f16_e64

v_ceil_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_ceil_f16_e64

v_cos_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_cos_f16_e64

v_cos_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_cos_f16_e64

v_cos_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_cos_f16_e64

v_cvt_f16_f32 v255.l, v1 quad_perm:[3,2,1,0]
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.l, v255 quad_perm:[3,2,1,0]
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, v1 quad_perm:[3,2,1,0]
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, v255 quad_perm:[3,2,1,0]
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_i16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_u16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_cvt_f16_u16_e64

v_cvt_f32_f16 v5, v199.l quad_perm:[3,2,1,0]
// GFX12: v_cvt_f32_f16_e64

v_cvt_f32_f16 v5, v199.h quad_perm:[3,2,1,0]
// GFX12: v_cvt_f32_f16_e64

v_cvt_i16_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_cvt_i16_f16_e64

v_cvt_i32_i16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_cvt_i32_i16_e64

v_cvt_norm_i16_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_u16_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_u16_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_cvt_u16_f16_e64

v_cvt_u32_u16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_cvt_u32_u16_e64

v_exp_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_exp_f16_e64

v_exp_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_exp_f16_e64

v_exp_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_exp_f16_e64

v_floor_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_floor_f16_e64

v_floor_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_floor_f16_e64

v_floor_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_floor_f16_e64

v_fract_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_fract_f16_e64

v_fract_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_fract_f16_e64

v_fract_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_fract_f16_e64

v_frexp_exp_i16_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_mant_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_frexp_mant_f16_e64

v_log_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_log_f16_e64

v_log_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_log_f16_e64

v_log_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_log_f16_e64

v_not_b16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_not_b16_e64

v_not_b16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_not_b16_e64

v_not_b16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_not_b16_e64

v_rcp_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_rcp_f16_e64

v_rcp_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_rcp_f16_e64

v_rndne_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_rndne_f16_e64

v_rndne_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_rndne_f16_e64

v_rsq_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_rsq_f16_e64

v_rsq_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_rsq_f16_e64

v_sat_pk_u8_i16 v199, v5 quad_perm:[3,2,1,0]
// GFX12: v_sat_pk_u8_i16_e64

v_sin_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_sin_f16_e64

v_sin_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_sin_f16_e64

v_sin_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_sin_f16_e64

v_sqrt_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_sqrt_f16_e64

v_trunc_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX12: v_trunc_f16_e64

v_trunc_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX12: v_trunc_f16_e64

v_ceil_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_ceil_f16_e64

v_ceil_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_ceil_f16_e64

v_ceil_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_ceil_f16_e64

v_cos_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cos_f16_e64

v_cos_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cos_f16_e64

v_cos_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cos_f16_e64

v_cvt_f16_f32 v255.l, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.l, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255.h, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_f16_f32_e64

v_cvt_f16_i16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_i16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_f16_i16_e64

v_cvt_f16_u16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_f16_u16_e64

v_cvt_f16_u16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_f16_u16_e64

v_cvt_f32_f16 v5, v199.l dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_f32_f16_e64

v_cvt_f32_f16 v5, v199.h dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_f32_f16_e64

v_cvt_i16_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_i16_f16_e64

v_cvt_i16_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_i16_f16_e64

v_cvt_i32_i16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_i32_i16_e64

v_cvt_norm_i16_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_norm_i16_f16_e64

v_cvt_norm_u16_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_norm_u16_f16_e64

v_cvt_u16_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_u16_f16_e64

v_cvt_u16_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_u16_f16_e64

v_cvt_u32_u16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_cvt_u32_u16_e64

v_exp_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_exp_f16_e64

v_exp_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_exp_f16_e64

v_exp_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_exp_f16_e64

v_floor_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_floor_f16_e64

v_floor_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_floor_f16_e64

v_floor_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_floor_f16_e64

v_fract_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_fract_f16_e64

v_fract_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_fract_f16_e64

v_fract_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_fract_f16_e64

v_frexp_exp_i16_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_frexp_exp_i16_f16_e64

v_frexp_mant_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_frexp_mant_f16_e64

v_frexp_mant_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_frexp_mant_f16_e64

v_log_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_log_f16_e64

v_log_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_log_f16_e64

v_log_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_log_f16_e64

v_not_b16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_not_b16_e64

v_not_b16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_not_b16_e64

v_not_b16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_not_b16_e64

v_rcp_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_rcp_f16_e64

v_rcp_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_rcp_f16_e64

v_rcp_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_rcp_f16_e64

v_rndne_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_rndne_f16_e64

v_rndne_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_rndne_f16_e64

v_rndne_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_rndne_f16_e64

v_rsq_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_rsq_f16_e64

v_rsq_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_rsq_f16_e64

v_rsq_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_rsq_f16_e64

v_sat_pk_u8_i16 v199, v5 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_sat_pk_u8_i16_e64

v_sin_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_sin_f16_e64

v_sin_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_sin_f16_e64

v_sin_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_sin_f16_e64

v_sqrt_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_sqrt_f16_e64

v_sqrt_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_sqrt_f16_e64

v_trunc_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_trunc_f16_e64

v_trunc_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_trunc_f16_e64

v_trunc_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_trunc_f16_e64
