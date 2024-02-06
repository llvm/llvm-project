// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck --check-prefix=GFX12 --implicit-check-not=error: %s

// check for error with sgpr or imm operands

v_dot4_f32_fp8_bf8 v0, s0, v2, v3 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_fp8_bf8 v0, v1, s0, v3 row_shr:15 row_mask:0x1 bank_mask:0x1 bound_ctrl:1 fi:1
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_bf8_fp8 v0, v1, v2, s0 row_shl:15
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_bf8_fp8 v0, 1.0, v2, v3 row_ror:15 row_mask:0x1 bank_mask:0x1 bound_ctrl:1 fi:1
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_fp8_fp8 v0, v1, 1.0, v3 row_mirror
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_fp8_fp8 v0, v1, v2, 1.0 row_half_mirror row_mask:0x1 bank_mask:0x1 bound_ctrl:1 fi:1
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_bf8_bf8 v0, v1, v2, 1 row_share:15
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
