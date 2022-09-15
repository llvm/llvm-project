// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX12-ERR --implicit-check-not=error: -strict-whitespace %s

s_prefetch_inst s[14:15], 0xffffff, m0, 7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 24-bit signed offset
// GFX12-ERR: s_prefetch_inst s[14:15], 0xffffff, m0, 7
// GFX12-ERR:                           ^

v_cubesc_f32_e64_dpp v5, v1, v2, 12345678 row_shr:4 row_mask:0xf bank_mask:0xf
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_add3_u32_e64_dpp v5, v1, v2, 49812340 dpp8:[7,6,5,4,3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_add3_u32_e64_dpp v5, v1, s1, v0 dpp8:[7,6,5,4,3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

; disallow space between colons
v_dual_mul_f32 v0, v0, v2 : : v_dual_mul_f32 v1, v1, v3
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: unknown token in expression

// On GFX12, v_dot8_i32_i4 is a valid SP3 alias for v_dot8_i32_iu4.
// However, we intentionally leave it unimplemented because on other
// processors v_dot8_i32_i4 denotes an instruction of a different
// behaviour, which is considered potentially dangerous.
v_dot8_i32_i4 v0, v1, v2, v3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

// On GFX12, v_dot4_i32_i8 is a valid SP3 alias for v_dot4_i32_iu8.
// However, we intentionally leave it unimplemented because on other
// processors v_dot4_i32_i8 denotes an instruction of a different
// behaviour, which is considered potentially dangerous.
v_dot4_i32_i8 v0, v1, v2, v3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4c_i32_i8 v0, v1, v2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fma_mix_f32_e64_dpp v5, s1, v3, v4 quad_perm:[3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_fma_mix_f32_e64_dpp v5, v1, s3, v4 quad_perm:[3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_fma_mix_f32_e64_dpp v5, s1, v3, v4 dpp8:[7,6,5,4,3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_fma_mix_f32_e64_dpp v5, v1, s3, v4 dpp8:[7,6,5,4,3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_fma_mixhi_f16_e64_dpp v5, v1, 0, v4 quad_perm:[3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_fma_mixlo_f16_e64_dpp v5, v1, 1, v4 dpp8:[7,6,5,4,3,2,1,0]
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
