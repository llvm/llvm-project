// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX1250-ERR --implicit-check-not=error: -strict-whitespace %s

s_load_b32 s4, s[2:3], 10 th:TH_LOAD_NT th:TH_LOAD_NT
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR: s_load_b32 s4, s[2:3], 10 th:TH_LOAD_NT th:TH_LOAD_NT
// GFX1250-ERR:                                         ^

s_load_b32 s4, s[2:3], 10 scope:SCOPE_SE scope:SCOPE_SE
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR: s_load_b32 s4, s[2:3], 10 scope:SCOPE_SE scope:SCOPE_SE
// GFX1250-ERR:                                          ^

s_load_b32 s4, s[2:3], 10 nv nv
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR: s_load_b32 s4, s[2:3], 10 nv nv
// GFX1250-ERR:                              ^

v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR: v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1250-ERR:                          ^

v_mov_b64 v[4:5], v[2:3] dpp8:[7,6,5,4,3,2,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR: v_mov_b64 v[4:5], v[2:3] dpp8:[7,6,5,4,3,2,1,0]
// GFX1250-ERR:                          ^

// For v_dual_cndmask_b32 use of the explicit src2 forces VOPD3 form even if it is vcc_lo.
// If src2 is omitted then it forces VOPD form. As a result a proper form of the instruction
// has to be used if the other component of the dual instruction cannot be used if that
// encoding.

v_dual_cndmask_b32 v2, v4, v1 :: v_dual_fma_f32 v7, v1, v2, v3
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid VOPDY instruction
// GFX1250-ERR: v_dual_cndmask_b32 v2, v4, v1 :: v_dual_fma_f32 v7, v1, v2, v3
// GFX1250-ERR:                                  ^

v_dual_fma_f32 v7, v1, v2, v3 :: v_dual_cndmask_b32 v2, v4, v1
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// GFX1250-ERR: v_dual_fma_f32 v7, v1, v2, v3 :: v_dual_cndmask_b32 v2, v4, v1
// GFX1250-ERR: ^

v_dual_cndmask_b32 v7, v1, v2 :: v_dual_cndmask_b32 v2, v4, v1, vcc_lo
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR: v_dual_cndmask_b32 v7, v1, v2 :: v_dual_cndmask_b32 v2, v4, v1, vcc_lo
// GFX1250-ERR:                                                                 ^

v_dual_cndmask_b32 v7, v1, v2, vcc_lo :: v_dual_cndmask_b32 v2, v4, v1
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// GFX1250-ERR: v_dual_cndmask_b32 v7, v1, v2, vcc_lo :: v_dual_cndmask_b32 v2, v4, v1
// GFX1250-ERR: ^

// Check for unique 64-bit literal

v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR: v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1250-ERR:                          ^

v_mov_b64 v[4:5], v[2:3] dpp8:[7,6,5,4,3,2,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR: v_mov_b64 v[4:5], v[2:3] dpp8:[7,6,5,4,3,2,1,0]
// GFX1250-ERR:                          ^

s_andn2_b64 s[2:3], 0x10abcdef12345678, 0xabcdef12345678
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX1250-ERR: s_andn2_b64 s[2:3], 0x10abcdef12345678, 0xabcdef12345678
// GFX1250-ERR: ^

s_bfe_u64 s[2:3], 0x10abcdef12345678, 100
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX1250-ERR: s_bfe_u64 s[2:3], 0x10abcdef12345678, 100
// GFX1250-ERR: ^

s_call_b64 s[2:3], 0x10abcdef12345678
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit signed jump offset
// GFX1250-ERR: s_call_b64 s[2:3], 0x10abcdef12345678
// GFX1250-ERR:                    ^

// VOP3 instructions cannot use 64-bit literals

v_ceil_f64_e64 v[254:255], 0x10abcdef12345678
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR: v_ceil_f64_e64 v[254:255], 0x10abcdef12345678
// GFX1250-ERR:                            ^

v_add_f64_e64 v[254:255], 0x10abcdef12345678, v[254:255]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR: v_add_f64_e64 v[254:255], 0x10abcdef12345678, v[254:255]
// GFX1250-ERR:                           ^

v_cmp_lt_f64_e64 vcc_lo, 0x10abcdef12345678, v[254:255]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR: v_cmp_lt_f64_e64 vcc_lo, 0x10abcdef12345678, v[254:255]
// GFX1250-ERR:                          ^

v_fma_f64 v[0:1], 0x10abcdef12345678, v[2:3], v[4:5]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR: v_fma_f64 v[0:1], 0x10abcdef12345678, v[2:3], v[4:5]
// GFX1250-ERR:                   ^

// Do not allow 64-bit literals for 32-bit operands. This may be possible to
// encode but not practically useful and can be misleading.

s_bfe_u64 s[2:3], 0x10abcdef12345678, 0x10abcdef12345678
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR: s_bfe_u64 s[2:3], 0x10abcdef12345678, 0x10abcdef12345678
// GFX1250-ERR:                                       ^

v_add_f32 v1, 0x12345678abcdef00, v2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR: v_add_f32 v1, 0x12345678abcdef00, v2
// GFX1250-ERR:               ^

v_ceil_f64 v[2:3], lit64(v[0:1])
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected immediate with lit modifier
// GFX1250-ERR: v_ceil_f64 v[2:3], lit64(v[0:1]
// GFX1250-ERR:                          ^

v_ceil_f64 v[2:3], lit64(123
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected closing parentheses
// GFX1250-ERR: v_ceil_f64 v[2:3], lit64(123
// GFX1250-ERR:                             ^

v_fmaak_f64 v[4:5], lit(lit64(0x7e8)), v[8:9], lit64(0x7e8)
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR: v_fmaak_f64 v[4:5], lit(lit64(0x7e8)), v[8:9], lit64(0x7e8)
// GFX1250-ERR:                              ^

v_fmaak_f64 v[4:5], lit64(lit64(0x7e8)), v[8:9], lit64(0x7e8)
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR: v_fmaak_f64 v[4:5], lit64(lit64(0x7e8)), v[8:9], lit64(0x7e8)
// GFX1250-ERR:                                ^

v_fmaak_f64 v[4:5], lit64(lit(0x7e8)), v[8:9], lit64(0x7e8)
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR: v_fmaak_f64 v[4:5], lit64(lit(0x7e8)), v[8:9], lit64(0x7e8)
// GFX1250-ERR:                              ^

v_fmamk_f64 v[4:5], 123.0, 123.1, v[6:7]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX1250-ERR: v_fmamk_f64 v[4:5], 123.0, 123.1, v[6:7]
// GFX1250-ERR:                     ^

v_fmamk_f64 v[4:5], 0x405ec00000000001, 123.0, v[6:7]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX1250-ERR: v_fmamk_f64 v[4:5], 0x405ec00000000001, 123.0, v[6:7]
// GFX1250-ERR:                     ^

v_fmaak_f64 v[4:5], 123.1, v[6:7], 123.0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX1250-ERR: v_fmaak_f64 v[4:5], 123.1, v[6:7], 123.0
// GFX1250-ERR:                     ^

v_fmaak_f64 v[4:5], 123.0, v[6:7], 0x405ec00000000001
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX1250-ERR: v_fmaak_f64 v[4:5], 123.0, v[6:7], 0x405ec00000000001
// GFX1250-ERR:                     ^

v_fmaak_f64 v[4:5], 0x7e8, v[8:9], lit64(0x7e8)
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX1250-ERR: v_fmaak_f64 v[4:5], 0x7e8, v[8:9], lit64(0x7e8)
// GFX1250-ERR:                     ^

v_pk_add_min_i16 v10, |v1|, v2, v3
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR: v_pk_add_min_i16 v10, |v1|, v2, v3
// GFX1250-ERR:                       ^

v_pk_add_min_i16 v10, -v1, v2, v3
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR: v_pk_add_min_i16 v10, -v1, v2, v3
// GFX1250-ERR:                       ^

v_pk_add_min_i16 v10, sext(v1), v2, v3
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR: v_pk_add_min_i16 v10, sext(v1), v2, v3
// GFX1250-ERR:                       ^

v_pk_add_min_i16 v10, v1, v2, v3 neg_lo:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR: v_pk_add_min_i16 v10, v1, v2, v3 neg_lo:[1,0,0]
// GFX1250-ERR:                                  ^

v_pk_add_min_i16 v10, v1, v2, v3 neg_hi:[1,0,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR: v_pk_add_min_i16 v10, v1, v2, v3 neg_hi:[1,0,0]
// GFX1250-ERR:                                  ^
