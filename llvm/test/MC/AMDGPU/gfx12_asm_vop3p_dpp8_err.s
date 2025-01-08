// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck --check-prefix=GFX12 --implicit-check-not=error: %s

// check for error with sgpr or imm operands

v_dot4_f32_fp8_bf8 v0, v1, v2, v3 dpp8:[0,1,2,3,4,5,6,7] row_mask:0x1
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_fp8_bf8 v0, s0, v2, v3 dpp8:[0,1,2,3,4,5,6,7]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_bf8_fp8 v0, v1, s0, v3 dpp8:[0,1,2,3,4,5,6,7]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_bf8_fp8 v0, v1, v2, s0 dpp8:[0,1,2,3,4,5,6,7]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_fp8_fp8 v0, 1.0, v2, v3 dpp8:[0,1,2,3,4,5,6,7]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_fp8_fp8 v0, v1, 1.0, v3 dpp8:[0,1,2,3,4,5,6,7]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_bf8_bf8 v0, v1, v2, 1.0 dpp8:[0,1,2,3,4,5,6,7]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_dot4_f32_bf8_bf8 v0, v1, v2, 1 dpp8:[0,1,2,3,4,5,6,7]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
