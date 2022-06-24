// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=error: %s

s_delay_alu
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction

s_delay_alu instid9(VALU_DEP_1)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid field name instid9

s_delay_alu instid0(VALU_DEP_9)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid value name VALU_DEP_9

s_delay_alu instid0(1)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: expected a value name

s_delay_alu instid0(VALU_DEP_9|)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: expected a right parenthesis

s_delay_alu instid0(VALU_DEP_1) | (SALU_CYCLE_1)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: expected a field name

s_delay_alu instid0(VALU_DEP_1) | SALU_CYCLE_1)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: expected a left parenthesis

lds_direct_load v15 wait_vdst:16
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

lds_direct_load v15 wait_vdst
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p10_f32 v0, v1, v2, v3 wait_exp:8
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_interp_p2_f32 v0, -v1, v2, v3 wait_exp
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

global_atomic_cmpswap_x2 v[1:4], v3, v[5:8], off offset:2047 glc
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cubesc_f32_e64_dpp v5, v1, v2, 12345678 row_shr:4 row_mask:0xf bank_mask:0xf
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_add3_u32_e64_dpp v5, v1, v2, 49812340 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_add3_u32_e64_dpp v5, v1, s1, v0 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cvt_f32_i32_e64_dpp v5, s1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cvt_f32_i32_e64_dpp v5, s1 row_shl:15 row_mask:0xf bank_mask:0xf
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cvt_f16_u16_e64_dpp v5, s1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_cvt_f16_u16_e64_dpp v5, s1 row_shl:1 row_mask:0xf bank_mask:0xf
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

; disallow space between colons
v_dual_mul_f32 v0, v0, v2 : : v_dual_mul_f32 v1, v1, v3
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: unknown token in expression

// On GFX11, v_dot8_i32_i4 is a valid SP3 alias for v_dot8_i32_iu4.
// However, we intentionally leave it unimplemented because on other
// processors v_dot8_i32_i4 denotes an instruction of a different
// behaviour, which is considered potentially dangerous.
v_dot8_i32_i4 v0, v1, v2, v3
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

// On GFX11, v_dot4_i32_i8 is a valid SP3 alias for v_dot4_i32_iu8.
// However, we intentionally leave it unimplemented because on other
// processors v_dot4_i32_i8 denotes an instruction of a different
// behaviour, which is considered potentially dangerous.
v_dot4_i32_i8 v0, v1, v2, v3
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot4c_i32_i8 v0, v1, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
