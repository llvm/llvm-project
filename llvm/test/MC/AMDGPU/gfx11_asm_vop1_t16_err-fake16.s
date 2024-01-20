// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=-real-true16,+wavefrontsize32,-wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=-real-true16,-wavefrontsize32,+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=error: %s

v_floor_f16_e32 v128, 0xfe0b
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_floor_f16_e32 v255, v1
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_floor_f16_e32 v5, v199
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_floor_f16_e32 v255, v1 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_floor_f16_e32 v5, v199 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_floor_f16_e32 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_floor_f16_e32 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_ceil_f16_e32 v128, 0xfe0b
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_ceil_f16_e32 v255, v1
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_ceil_f16_e32 v5, v199
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_ceil_f16_e32 v255, v1 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_ceil_f16_e32 v5, v199 quad_perm:[3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_ceil_f16_e32 v255, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_ceil_f16_e32 v5, v199 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
