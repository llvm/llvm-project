// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX1250-ERR --implicit-check-not=error: -strict-whitespace %s

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
