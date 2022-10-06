// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefix=GFX11 --implicit-check-not=_e32 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck --check-prefix=GFX11 --implicit-check-not=_e32 %s

v_ceil_f16 v128, 0xfe0b
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, -1
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, 0.5
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, exec_hi
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, exec_lo
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, m0
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, null
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, s1
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, s105
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, src_scc
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, ttmp15
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, v1
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, v127
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, vcc_hi
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, vcc_lo
// GFX11: v_ceil_f16_e64

v_ceil_f16 v5, v199
// GFX11: v_ceil_f16_e64

v_cos_f16 v128, 0xfe0b
// GFX11: v_cos_f16_e64

v_cos_f16 v255, -1
// GFX11: v_cos_f16_e64

v_cos_f16 v255, 0.5
// GFX11: v_cos_f16_e64

v_cos_f16 v255, exec_hi
// GFX11: v_cos_f16_e64

v_cos_f16 v255, exec_lo
// GFX11: v_cos_f16_e64

v_cos_f16 v255, m0
// GFX11: v_cos_f16_e64

v_cos_f16 v255, null
// GFX11: v_cos_f16_e64

v_cos_f16 v255, s1
// GFX11: v_cos_f16_e64

v_cos_f16 v255, s105
// GFX11: v_cos_f16_e64

v_cos_f16 v255, src_scc
// GFX11: v_cos_f16_e64

v_cos_f16 v255, ttmp15
// GFX11: v_cos_f16_e64

v_cos_f16 v255, v1
// GFX11: v_cos_f16_e64

v_cos_f16 v255, v127
// GFX11: v_cos_f16_e64

v_cos_f16 v255, vcc_hi
// GFX11: v_cos_f16_e64

v_cos_f16 v255, vcc_lo
// GFX11: v_cos_f16_e64

v_cos_f16 v5, v199
// GFX11: v_cos_f16_e64

v_cvt_f16_f32 v128, 0xaf123456
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, -1
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, 0.5
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, exec_hi
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, exec_lo
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, m0
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, null
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, s1
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, s105
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, src_scc
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, ttmp15
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, v1
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, v255
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, vcc_hi
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, vcc_lo
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_i16 v128, 0xfe0b
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, -1
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, 0.5
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, exec_hi
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, exec_lo
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, m0
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, null
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, s1
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, s105
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, src_scc
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, ttmp15
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, v1
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, v127
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, vcc_hi
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, vcc_lo
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v5, v199
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_u16 v128, 0xfe0b
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, -1
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, 0.5
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, exec_hi
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, exec_lo
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, m0
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, null
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, s1
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, s105
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, src_scc
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, ttmp15
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, v1
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, v127
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, vcc_hi
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, vcc_lo
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v5, v199
// GFX11: v_cvt_f16_u16_e64

v_cvt_f32_f16 v5, v199
// GFX11: v_cvt_f32_f16_e64

v_cvt_i16_f16 v128, 0xfe0b
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, -1
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, 0.5
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, exec_hi
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, exec_lo
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, m0
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, null
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, s1
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, s105
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, src_scc
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, ttmp15
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, v1
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, v127
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, vcc_hi
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, vcc_lo
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v5, v199
// GFX11: v_cvt_i16_f16_e64

v_cvt_i32_i16 v5, v199
// GFX11: v_cvt_i32_i16_e64

v_cvt_norm_i16_f16 v128, 0xfe0b
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, -1
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, 0.5
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, exec_hi
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, exec_lo
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, m0
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, null
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, s1
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, s105
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, src_scc
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, ttmp15
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, v1
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, v127
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, vcc_hi
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, vcc_lo
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v5, v199
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_u16_f16 v128, 0xfe0b
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, -1
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, 0.5
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, exec_hi
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, exec_lo
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, m0
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, null
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, s1
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, s105
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, src_scc
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, ttmp15
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, v1
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, v127
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, vcc_hi
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, vcc_lo
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v5, v199
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_u16_f16 v128, 0xfe0b
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, -1
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, 0.5
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, exec_hi
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, exec_lo
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, m0
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, null
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, s1
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, s105
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, src_scc
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, ttmp15
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, v1
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, v127
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, vcc_hi
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, vcc_lo
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v5, v199
// GFX11: v_cvt_u16_f16_e64

v_cvt_u32_u16 v5, v199
// GFX11: v_cvt_u32_u16_e64

v_exp_f16 v128, 0xfe0b
// GFX11: v_exp_f16_e64

v_exp_f16 v255, -1
// GFX11: v_exp_f16_e64

v_exp_f16 v255, 0.5
// GFX11: v_exp_f16_e64

v_exp_f16 v255, exec_hi
// GFX11: v_exp_f16_e64

v_exp_f16 v255, exec_lo
// GFX11: v_exp_f16_e64

v_exp_f16 v255, m0
// GFX11: v_exp_f16_e64

v_exp_f16 v255, null
// GFX11: v_exp_f16_e64

v_exp_f16 v255, s1
// GFX11: v_exp_f16_e64

v_exp_f16 v255, s105
// GFX11: v_exp_f16_e64

v_exp_f16 v255, src_scc
// GFX11: v_exp_f16_e64

v_exp_f16 v255, ttmp15
// GFX11: v_exp_f16_e64

v_exp_f16 v255, v1
// GFX11: v_exp_f16_e64

v_exp_f16 v255, v127
// GFX11: v_exp_f16_e64

v_exp_f16 v255, vcc_hi
// GFX11: v_exp_f16_e64

v_exp_f16 v255, vcc_lo
// GFX11: v_exp_f16_e64

v_exp_f16 v5, v199
// GFX11: v_exp_f16_e64

v_floor_f16 v128, 0xfe0b
// GFX11: v_floor_f16_e64

v_floor_f16 v255, -1
// GFX11: v_floor_f16_e64

v_floor_f16 v255, 0.5
// GFX11: v_floor_f16_e64

v_floor_f16 v255, exec_hi
// GFX11: v_floor_f16_e64

v_floor_f16 v255, exec_lo
// GFX11: v_floor_f16_e64

v_floor_f16 v255, m0
// GFX11: v_floor_f16_e64

v_floor_f16 v255, null
// GFX11: v_floor_f16_e64

v_floor_f16 v255, s1
// GFX11: v_floor_f16_e64

v_floor_f16 v255, s105
// GFX11: v_floor_f16_e64

v_floor_f16 v255, src_scc
// GFX11: v_floor_f16_e64

v_floor_f16 v255, ttmp15
// GFX11: v_floor_f16_e64

v_floor_f16 v255, v1
// GFX11: v_floor_f16_e64

v_floor_f16 v255, v127
// GFX11: v_floor_f16_e64

v_floor_f16 v255, vcc_hi
// GFX11: v_floor_f16_e64

v_floor_f16 v255, vcc_lo
// GFX11: v_floor_f16_e64

v_floor_f16 v5, v199
// GFX11: v_floor_f16_e64

v_fract_f16 v128, 0xfe0b
// GFX11: v_fract_f16_e64

v_fract_f16 v255, -1
// GFX11: v_fract_f16_e64

v_fract_f16 v255, 0.5
// GFX11: v_fract_f16_e64

v_fract_f16 v255, exec_hi
// GFX11: v_fract_f16_e64

v_fract_f16 v255, exec_lo
// GFX11: v_fract_f16_e64

v_fract_f16 v255, m0
// GFX11: v_fract_f16_e64

v_fract_f16 v255, null
// GFX11: v_fract_f16_e64

v_fract_f16 v255, s1
// GFX11: v_fract_f16_e64

v_fract_f16 v255, s105
// GFX11: v_fract_f16_e64

v_fract_f16 v255, src_scc
// GFX11: v_fract_f16_e64

v_fract_f16 v255, ttmp15
// GFX11: v_fract_f16_e64

v_fract_f16 v255, v1
// GFX11: v_fract_f16_e64

v_fract_f16 v255, v127
// GFX11: v_fract_f16_e64

v_fract_f16 v255, vcc_hi
// GFX11: v_fract_f16_e64

v_fract_f16 v255, vcc_lo
// GFX11: v_fract_f16_e64

v_fract_f16 v5, v199
// GFX11: v_fract_f16_e64

v_frexp_exp_i16_f16 v128, 0xfe0b
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, -1
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, 0.5
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, exec_hi
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, exec_lo
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, m0
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, null
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, s1
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, s105
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, src_scc
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, ttmp15
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, v1
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, v127
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, vcc_hi
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, vcc_lo
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v5, v199
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_mant_f16 v128, 0xfe0b
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, -1
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, 0.5
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, exec_hi
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, exec_lo
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, m0
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, null
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, s1
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, s105
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, src_scc
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, ttmp15
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, v1
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, v127
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, vcc_hi
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, vcc_lo
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v5, v199
// GFX11: v_frexp_mant_f16_e64

v_log_f16 v128, 0xfe0b
// GFX11: v_log_f16_e64

v_log_f16 v255, -1
// GFX11: v_log_f16_e64

v_log_f16 v255, 0.5
// GFX11: v_log_f16_e64

v_log_f16 v255, exec_hi
// GFX11: v_log_f16_e64

v_log_f16 v255, exec_lo
// GFX11: v_log_f16_e64

v_log_f16 v255, m0
// GFX11: v_log_f16_e64

v_log_f16 v255, null
// GFX11: v_log_f16_e64

v_log_f16 v255, s1
// GFX11: v_log_f16_e64

v_log_f16 v255, s105
// GFX11: v_log_f16_e64

v_log_f16 v255, src_scc
// GFX11: v_log_f16_e64

v_log_f16 v255, ttmp15
// GFX11: v_log_f16_e64

v_log_f16 v255, v1
// GFX11: v_log_f16_e64

v_log_f16 v255, v127
// GFX11: v_log_f16_e64

v_log_f16 v255, vcc_hi
// GFX11: v_log_f16_e64

v_log_f16 v255, vcc_lo
// GFX11: v_log_f16_e64

v_log_f16 v5, v199
// GFX11: v_log_f16_e64

v_not_b16 v128, 0xfe0b
// GFX11: v_not_b16_e64

v_not_b16 v255, -1
// GFX11: v_not_b16_e64

v_not_b16 v255, 0.5
// GFX11: v_not_b16_e64

v_not_b16 v255, exec_hi
// GFX11: v_not_b16_e64

v_not_b16 v255, exec_lo
// GFX11: v_not_b16_e64

v_not_b16 v255, m0
// GFX11: v_not_b16_e64

v_not_b16 v255, null
// GFX11: v_not_b16_e64

v_not_b16 v255, s1
// GFX11: v_not_b16_e64

v_not_b16 v255, s105
// GFX11: v_not_b16_e64

v_not_b16 v255, src_scc
// GFX11: v_not_b16_e64

v_not_b16 v255, ttmp15
// GFX11: v_not_b16_e64

v_not_b16 v255, v1
// GFX11: v_not_b16_e64

v_not_b16 v255, v127
// GFX11: v_not_b16_e64

v_not_b16 v255, vcc_hi
// GFX11: v_not_b16_e64

v_not_b16 v255, vcc_lo
// GFX11: v_not_b16_e64

v_not_b16 v5, v199
// GFX11: v_not_b16_e64

v_rcp_f16 v128, 0xfe0b
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, -1
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, 0.5
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, exec_hi
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, exec_lo
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, m0
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, null
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, s1
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, s105
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, src_scc
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, ttmp15
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, v1
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, v127
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, vcc_hi
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, vcc_lo
// GFX11: v_rcp_f16_e64

v_rcp_f16 v5, v199
// GFX11: v_rcp_f16_e64

v_rndne_f16 v128, 0xfe0b
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, -1
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, 0.5
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, exec_hi
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, exec_lo
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, m0
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, null
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, s1
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, s105
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, src_scc
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, ttmp15
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, v1
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, v127
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, vcc_hi
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, vcc_lo
// GFX11: v_rndne_f16_e64

v_rndne_f16 v5, v199
// GFX11: v_rndne_f16_e64

v_rsq_f16 v128, 0xfe0b
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, -1
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, 0.5
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, exec_hi
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, exec_lo
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, m0
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, null
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, s1
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, s105
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, src_scc
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, ttmp15
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, v1
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, v127
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, vcc_hi
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, vcc_lo
// GFX11: v_rsq_f16_e64

v_rsq_f16 v5, v199
// GFX11: v_rsq_f16_e64

v_sin_f16 v128, 0xfe0b
// GFX11: v_sin_f16_e64

v_sin_f16 v255, -1
// GFX11: v_sin_f16_e64

v_sin_f16 v255, 0.5
// GFX11: v_sin_f16_e64

v_sin_f16 v255, exec_hi
// GFX11: v_sin_f16_e64

v_sin_f16 v255, exec_lo
// GFX11: v_sin_f16_e64

v_sin_f16 v255, m0
// GFX11: v_sin_f16_e64

v_sin_f16 v255, null
// GFX11: v_sin_f16_e64

v_sin_f16 v255, s1
// GFX11: v_sin_f16_e64

v_sin_f16 v255, s105
// GFX11: v_sin_f16_e64

v_sin_f16 v255, src_scc
// GFX11: v_sin_f16_e64

v_sin_f16 v255, ttmp15
// GFX11: v_sin_f16_e64

v_sin_f16 v255, v1
// GFX11: v_sin_f16_e64

v_sin_f16 v255, v127
// GFX11: v_sin_f16_e64

v_sin_f16 v255, vcc_hi
// GFX11: v_sin_f16_e64

v_sin_f16 v255, vcc_lo
// GFX11: v_sin_f16_e64

v_sin_f16 v5, v199
// GFX11: v_sin_f16_e64

v_sqrt_f16 v128, 0xfe0b
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, -1
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, 0.5
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, exec_hi
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, exec_lo
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, m0
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, null
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, s1
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, s105
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, src_scc
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, ttmp15
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, v1
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, v127
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, vcc_hi
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, vcc_lo
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v5, v199
// GFX11: v_sqrt_f16_e64

v_trunc_f16 v128, 0xfe0b
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, -1
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, 0.5
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, exec_hi
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, exec_lo
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, m0
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, null
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, s1
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, s105
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, src_scc
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, ttmp15
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, v1
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, v127
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, vcc_hi
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, vcc_lo
// GFX11: v_trunc_f16_e64

v_trunc_f16 v5, v199
// GFX11: v_trunc_f16_e64

v_ceil_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_ceil_f16_e64

v_ceil_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_ceil_f16_e64

v_cos_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_cos_f16_e64

v_cos_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_cos_f16_e64

v_cos_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_cos_f16_e64

v_cvt_f16_f32 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, v255 quad_perm:[3,2,1,0]
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_i16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_u16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_cvt_f16_u16_e64

v_cvt_f32_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_cvt_f32_f16_e64

v_cvt_i16_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_cvt_i16_f16_e64

v_cvt_i32_i16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_cvt_i32_i16_e64

v_cvt_norm_i16_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_u16_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_u16_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_cvt_u16_f16_e64

v_cvt_u32_u16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_cvt_u32_u16_e64

v_exp_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_exp_f16_e64

v_exp_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_exp_f16_e64

v_exp_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_exp_f16_e64

v_floor_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_floor_f16_e64

v_floor_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_floor_f16_e64

v_floor_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_floor_f16_e64

v_fract_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_fract_f16_e64

v_fract_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_fract_f16_e64

v_fract_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_fract_f16_e64

v_frexp_exp_i16_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_mant_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_frexp_mant_f16_e64

v_log_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_log_f16_e64

v_log_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_log_f16_e64

v_log_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_log_f16_e64

v_not_b16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_not_b16_e64

v_not_b16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_not_b16_e64

v_not_b16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_not_b16_e64

v_rcp_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_rcp_f16_e64

v_rcp_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_rcp_f16_e64

v_rndne_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_rndne_f16_e64

v_rndne_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_rndne_f16_e64

v_rsq_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_rsq_f16_e64

v_rsq_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_rsq_f16_e64

v_sin_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_sin_f16_e64

v_sin_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_sin_f16_e64

v_sin_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_sin_f16_e64

v_sqrt_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_sqrt_f16_e64

v_trunc_f16 v255, v1 quad_perm:[3,2,1,0]
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, v127 quad_perm:[3,2,1,0]
// GFX11: v_trunc_f16_e64

v_trunc_f16 v5, v199 quad_perm:[3,2,1,0]
// GFX11: v_trunc_f16_e64

v_ceil_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_ceil_f16_e64

v_ceil_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_ceil_f16_e64

v_ceil_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_ceil_f16_e64

v_cos_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cos_f16_e64

v_cos_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cos_f16_e64

v_cos_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cos_f16_e64

v_cvt_f16_f32 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_f32 v255, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_f16_f32_e64

v_cvt_f16_i16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_i16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_f16_i16_e64

v_cvt_f16_u16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_f16_u16_e64

v_cvt_f16_u16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_f16_u16_e64

v_cvt_f32_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_f32_f16_e64

v_cvt_i16_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_i16_f16_e64

v_cvt_i16_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_i16_f16_e64

v_cvt_i32_i16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_i32_i16_e64

v_cvt_norm_i16_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_i16_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_norm_i16_f16_e64

v_cvt_norm_u16_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_norm_u16_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_norm_u16_f16_e64

v_cvt_u16_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_u16_f16_e64

v_cvt_u16_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_u16_f16_e64

v_cvt_u32_u16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cvt_u32_u16_e64

v_exp_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_exp_f16_e64

v_exp_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_exp_f16_e64

v_exp_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_exp_f16_e64

v_floor_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_floor_f16_e64

v_floor_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_floor_f16_e64

v_floor_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_floor_f16_e64

v_fract_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_fract_f16_e64

v_fract_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_fract_f16_e64

v_fract_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_fract_f16_e64

v_frexp_exp_i16_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_exp_i16_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_frexp_exp_i16_f16_e64

v_frexp_mant_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_frexp_mant_f16_e64

v_frexp_mant_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_frexp_mant_f16_e64

v_log_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_log_f16_e64

v_log_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_log_f16_e64

v_log_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_log_f16_e64

v_not_b16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_not_b16_e64

v_not_b16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_not_b16_e64

v_not_b16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_not_b16_e64

v_rcp_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_rcp_f16_e64

v_rcp_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_rcp_f16_e64

v_rcp_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_rcp_f16_e64

v_rndne_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_rndne_f16_e64

v_rndne_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_rndne_f16_e64

v_rndne_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_rndne_f16_e64

v_rsq_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_rsq_f16_e64

v_rsq_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_rsq_f16_e64

v_rsq_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_rsq_f16_e64

v_sin_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_sin_f16_e64

v_sin_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_sin_f16_e64

v_sin_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_sin_f16_e64

v_sqrt_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_sqrt_f16_e64

v_sqrt_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_sqrt_f16_e64

v_trunc_f16 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_trunc_f16_e64

v_trunc_f16 v255, v127 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_trunc_f16_e64

v_trunc_f16 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_trunc_f16_e64

