// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -mattr=-real=true16 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX13 --implicit-check-not=error %s

v_floor_f16_e32 v128, 0xfe0b
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_floor_f16_e32 v255, v1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_floor_f16_e32 v5, v199
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_floor_f16_e32 v255, v1 quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_floor_f16_e32 v5, v199 quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_floor_f16_e32 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_floor_f16_e32 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_ceil_f16_e32 v128, 0xfe0b
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_ceil_f16_e32 v255, v1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_ceil_f16_e32 v5, v199
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_ceil_f16_e32 v255, v1 quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_ceil_f16_e32 v5, v199 quad_perm:[3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_ceil_f16_e32 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_ceil_f16_e32 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_rcp_f16_e32 v128, 0xfe0b
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_rcp_f16_e32 v255, v1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_rcp_f16_e32 v5, v199
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sqrt_f16_e32 v128, 0xfe0b
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sqrt_f16_e32 v255, v1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sqrt_f16_e32 v5, v199
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_rsq_f16_e32 v128, 0xfe0b
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_rsq_f16_e32 v255, v1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_rsq_f16_e32 v5, v199
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_log_f16_e32 v128, 0xfe0b
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_log_f16_e32 v255, v1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_log_f16_e32 v5, v199
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_exp_f16_e32 v128, 0xfe0b
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_exp_f16_e32 v255, v1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_exp_f16_e32 v5, v199
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
