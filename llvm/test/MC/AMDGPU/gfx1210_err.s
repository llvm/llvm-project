// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX1210-ERR --implicit-check-not=error: -strict-whitespace %s

s_set_vgpr_msb -1
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: s_set_vgpr_msb accepts values in range [0..15]
// GFX1210-ERR: s_set_vgpr_msb -1
// GFX1210-ERR:                ^

s_set_vgpr_msb 16
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: s_set_vgpr_msb accepts values in range [0..15]
// GFX1210-ERR: s_set_vgpr_msb 16
// GFX1210-ERR:                ^

s_load_b32 s4, s[2:3], 10 th:TH_LOAD_NT th:TH_LOAD_NT
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: s_load_b32 s4, s[2:3], 10 th:TH_LOAD_NT th:TH_LOAD_NT
// GFX1210-ERR:                                         ^

s_load_b32 s4, s[2:3], 10 scope:SCOPE_SE scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: s_load_b32 s4, s[2:3], 10 scope:SCOPE_SE scope:SCOPE_SE
// GFX1210-ERR:                                          ^

s_load_b32 s4, s[2:3], 10 nv nv
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: s_load_b32 s4, s[2:3], 10 nv nv
// GFX1210-ERR:                              ^

v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1210-ERR: v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1210-ERR:                          ^

// For v_dual_cndmask_b32 use of the explicit src2 forces VOPD3 form even if it is vcc_lo.
// If src2 is omitted then it forces VOPD form. As a result a proper form of the instruction
// has to be used if the other component of the dual instruction cannot be used if that
// encoding.

v_dual_cndmask_b32 v2, v4, v1 :: v_dual_fma_f32 v7, v1, v2, v3
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid VOPDY instruction
// GFX1210-ERR: v_dual_cndmask_b32 v2, v4, v1 :: v_dual_fma_f32 v7, v1, v2, v3
// GFX1210-ERR:                                  ^

v_dual_fma_f32 v7, v1, v2, v3 :: v_dual_cndmask_b32 v2, v4, v1
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// GFX1210-ERR: v_dual_fma_f32 v7, v1, v2, v3 :: v_dual_cndmask_b32 v2, v4, v1
// GFX1210-ERR: ^

v_dual_cndmask_b32 v7, v1, v2 :: v_dual_cndmask_b32 v2, v4, v1, vcc_lo
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: v_dual_cndmask_b32 v7, v1, v2 :: v_dual_cndmask_b32 v2, v4, v1, vcc_lo
// GFX1210-ERR:                                                                 ^

v_dual_cndmask_b32 v7, v1, v2, vcc_lo :: v_dual_cndmask_b32 v2, v4, v1
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// GFX1210-ERR: v_dual_cndmask_b32 v7, v1, v2, vcc_lo :: v_dual_cndmask_b32 v2, v4, v1
// GFX1210-ERR: ^

// Check for unique 64-bit literal

s_andn2_b64 s[2:3], 0x10abcdef12345678, 0xabcdef12345678
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX1210-ERR: s_andn2_b64 s[2:3], 0x10abcdef12345678, 0xabcdef12345678
// GFX1210-ERR: ^

s_bfe_u64 s[2:3], 0x10abcdef12345678, 100
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX1210-ERR: s_bfe_u64 s[2:3], 0x10abcdef12345678, 100
// GFX1210-ERR: ^

s_call_b64 s[2:3], 0x10abcdef12345678
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit signed jump offset
// GFX1210-ERR: s_call_b64 s[2:3], 0x10abcdef12345678
// GFX1210-ERR:                    ^

// VOP3 instructions cannot use 64-bit literals

v_ceil_f64_e64 v[254:255], 0x10abcdef12345678
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: v_ceil_f64_e64 v[254:255], 0x10abcdef12345678
// GFX1210-ERR:                            ^

v_add_f64_e64 v[254:255], 0x10abcdef12345678, v[254:255]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: v_add_f64_e64 v[254:255], 0x10abcdef12345678, v[254:255]
// GFX1210-ERR:                           ^

v_cmp_lt_f64_e64 vcc_lo, 0x10abcdef12345678, v[254:255]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: v_cmp_lt_f64_e64 vcc_lo, 0x10abcdef12345678, v[254:255]
// GFX1210-ERR:                          ^

v_fma_f64 v[0:1], 0x10abcdef12345678, v[2:3], v[4:5]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: v_fma_f64 v[0:1], 0x10abcdef12345678, v[2:3], v[4:5]
// GFX1210-ERR:                   ^

// Do not allow 64-bit literals for 32-bit operands. This may be possible to
// encode but not practically useful and can be misleading.

s_bfe_u64 s[2:3], 0x10abcdef12345678, 0x10abcdef12345678
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: s_bfe_u64 s[2:3], 0x10abcdef12345678, 0x10abcdef12345678
// GFX1210-ERR:                                       ^

v_add_f32 v1, 0x12345678abcdef00, v2
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: v_add_f32 v1, 0x12345678abcdef00, v2
// GFX1210-ERR:               ^

v_ceil_f64 v[2:3], lit64(v[0:1])
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected immediate with lit modifier
// GFX1210-ERR: v_ceil_f64 v[2:3], lit64(v[0:1]
// GFX1210-ERR:                          ^

v_ceil_f64 v[2:3], lit64(123
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected closing parentheses
// GFX1210-ERR: v_ceil_f64 v[2:3], lit64(123
// GFX1210-ERR:                             ^
