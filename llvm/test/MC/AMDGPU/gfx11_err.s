// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=error: %s

s_delay_alu
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction

s_delay_alu instid9(VALU_DEP_1)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid field name instid9

s_delay_alu instid0(VALU_DEP_9)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid value name VALU_DEP_9

s_delay_alu instid0(1)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: expected a value name

v_cubesc_f32_e64_dpp v5, v1, v2, 12345678 row_shr:4 row_mask:0xf bank_mask:0xf
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_add3_u32_e64_dpp v5, v1, v2, 49812340 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_add3_u32_e64_dpp v5, v1, s1, v0 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

; disallow space between colons
v_dual_mul_f32 v0, v0, v2 : : v_dual_mul_f32 v1, v1, v3
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: unknown token in expression
