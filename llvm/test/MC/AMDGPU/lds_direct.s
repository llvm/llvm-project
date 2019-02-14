// RUN: llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck %s --check-prefix=GFX9

//---------------------------------------------------------------------------//
// VOP1/3
//---------------------------------------------------------------------------//

v_mov_b32 v0, src_lds_direct
// GFX9: v_mov_b32_e32 v0, src_lds_direct ; encoding: [0xfe,0x02,0x00,0x7e]

v_mov_b32_e64 v0, src_lds_direct
// GFX9: v_mov_b32_e64 v0, src_lds_direct ; encoding: [0x00,0x00,0x41,0xd1,0xfe,0x00,0x00,0x00]

v_cvt_f64_i32 v[0:1], src_lds_direct
// GFX9: v_cvt_f64_i32_e32 v[0:1], src_lds_direct ; encoding: [0xfe,0x08,0x00,0x7e]

v_cvt_f64_i32_e64 v[0:1], src_lds_direct
// GFX9: v_cvt_f64_i32_e64 v[0:1], src_lds_direct ; encoding: [0x00,0x00,0x44,0xd1,0xfe,0x00,0x00,0x00]

v_mov_fed_b32 v0, src_lds_direct
// GFX9: v_mov_fed_b32_e32 v0, src_lds_direct ; encoding: [0xfe,0x12,0x00,0x7e]

v_mov_fed_b32_e64 v0, src_lds_direct
// GFX9: v_mov_fed_b32_e64 v0, src_lds_direct ; encoding: [0x00,0x00,0x49,0xd1,0xfe,0x00,0x00,0x00]

v_fract_f32 v0, src_lds_direct
// GFX9: v_fract_f32_e32 v0, src_lds_direct ; encoding: [0xfe,0x36,0x00,0x7e]

v_fract_f32_e64 v0, src_lds_direct
// GFX9: v_fract_f32_e64 v0, src_lds_direct ; encoding: [0x00,0x00,0x5b,0xd1,0xfe,0x00,0x00,0x00]

v_cvt_f16_u16 v0, src_lds_direct
// GFX9: v_cvt_f16_u16_e32 v0, src_lds_direct ; encoding: [0xfe,0x72,0x00,0x7e]

//---------------------------------------------------------------------------//
// VOP2/3
//---------------------------------------------------------------------------//

v_cndmask_b32 v0, src_lds_direct, v0, vcc
// GFX9: v_cndmask_b32_e32 v0, src_lds_direct, v0, vcc ; encoding: [0xfe,0x00,0x00,0x00]

v_cndmask_b32_e64 v0, src_lds_direct, v0, s[0:1]
// GFX9: v_cndmask_b32_e64 v0, src_lds_direct, v0, s[0:1] ; encoding: [0x00,0x00,0x00,0xd1,0xfe,0x00,0x02,0x00]

v_add_f32 v0, src_lds_direct, v0
// GFX9: v_add_f32_e32 v0, src_lds_direct, v0 ; encoding: [0xfe,0x00,0x00,0x02]

v_add_f32_e64 v0, src_lds_direct, v0
// GFX9: v_add_f32_e64 v0, src_lds_direct, v0 ; encoding: [0x00,0x00,0x01,0xd1,0xfe,0x00,0x02,0x00]

v_mul_i32_i24 v0, src_lds_direct, v0
// GFX9: v_mul_i32_i24_e32 v0, src_lds_direct, v0 ; encoding: [0xfe,0x00,0x00,0x0c]

v_add_co_u32 v0, vcc, src_lds_direct, v0
// GFX9: v_add_co_u32_e32 v0, vcc, src_lds_direct, v0 ; encoding: [0xfe,0x00,0x00,0x32]

//---------------------------------------------------------------------------//
// VOP3
//---------------------------------------------------------------------------//

v_add_co_u32_e64 v0, s[0:1], src_lds_direct, v0
// GFX9: v_add_co_u32_e64 v0, s[0:1], src_lds_direct, v0 ; encoding: [0x00,0x00,0x19,0xd1,0xfe,0x00,0x02,0x00]

v_madmk_f16 v0, src_lds_direct, 0x1121, v0
// GFX9: v_madmk_f16 v0, src_lds_direct, 0x1121, v0 ; encoding: [0xfe,0x00,0x00,0x48,0x21,0x11,0x00,0x00]

v_madak_f16 v0, src_lds_direct, v0, 0x1121
// GFX9: v_madak_f16 v0, src_lds_direct, v0, 0x1121 ; encoding: [0xfe,0x00,0x00,0x4a,0x21,0x11,0x00,0x00]

v_mad_f32 v0, src_lds_direct, v0, v0
// GFX9: v_mad_f32 v0, src_lds_direct, v0, v0 ; encoding: [0x00,0x00,0xc1,0xd1,0xfe,0x00,0x02,0x04]

v_fma_f32 v0, src_lds_direct, v0, v0
// GFX9: v_fma_f32 v0, src_lds_direct, v0, v0 ; encoding: [0x00,0x00,0xcb,0xd1,0xfe,0x00,0x02,0x04]

v_min3_i16 v0, src_lds_direct, v0, v0
// GFX9: v_min3_i16 v0, src_lds_direct, v0, v0 ; encoding: [0x00,0x00,0xf5,0xd1,0xfe,0x00,0x02,0x04]

v_max3_f16 v0, src_lds_direct, v0, v0
// GFX9: v_max3_f16 v0, src_lds_direct, v0, v0 ; encoding: [0x00,0x00,0xf7,0xd1,0xfe,0x00,0x02,0x04]

//---------------------------------------------------------------------------//
// VOP3P
//---------------------------------------------------------------------------//

v_pk_mad_i16 v0, src_lds_direct, v0, v0
// GFX9: v_pk_mad_i16 v0, src_lds_direct, v0, v0 ; encoding: [0x00,0x40,0x80,0xd3,0xfe,0x00,0x02,0x1c]

v_pk_add_i16 v0, src_lds_direct, v0
// GFX9: v_pk_add_i16 v0, src_lds_direct, v0 ; encoding: [0x00,0x00,0x82,0xd3,0xfe,0x00,0x02,0x18]

//---------------------------------------------------------------------------//
// VOPC
//---------------------------------------------------------------------------//

v_cmp_lt_f16 vcc, src_lds_direct, v0
// GFX9: v_cmp_lt_f16_e32 vcc, src_lds_direct, v0 ; encoding: [0xfe,0x00,0x42,0x7c]

v_cmp_eq_f32 vcc, src_lds_direct, v0
// GFX9: v_cmp_eq_f32_e32 vcc, src_lds_direct, v0 ; encoding: [0xfe,0x00,0x84,0x7c]

v_cmpx_neq_f32 vcc, src_lds_direct, v0
// GFX9: v_cmpx_neq_f32_e32 vcc, src_lds_direct, v0 ; encoding: [0xfe,0x00,0xba,0x7c]

//---------------------------------------------------------------------------//
// lds_direct alias
//---------------------------------------------------------------------------//

v_cmp_lt_f16 vcc, lds_direct, v0
// GFX9: v_cmp_lt_f16_e32 vcc, src_lds_direct, v0 ; encoding: [0xfe,0x00,0x42,0x7c]

//---------------------------------------------------------------------------//
// FIXME: enable lds_direct for the following opcodes and add tests
//---------------------------------------------------------------------------//

//v_readfirstlane_b32 s0, src_lds_direct
//v_readlane_b32 s0, src_lds_direct, s0
