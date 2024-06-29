// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12 --implicit-check-not=error %s

v_add_f16_e32 v255, v1, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmaak_f16_e32 v255, v1, v2, 0xfe0b
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmac_f16_e32 v255, v1, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmamk_f16_e32 v255, v1, 0xfe0b, v3
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_ldexp_f16_e32 v255, v1, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_max_num_f16_e32 v255, v1, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_min_num_f16_e32 v255, v1, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_mul_f16_e32 v255, v1, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_f16_e32 v255, v1, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_f16_e32 v255, v1, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_f16_e32 v5, v255, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmaak_f16_e32 v5, v255, v2, 0xfe0b
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmac_f16_e32 v5, v255, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmamk_f16_e32 v5, v255, 0xfe0b, v3
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_ldexp_f16_e32 v5, v255, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_max_num_f16_e32 v5, v255, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_min_num_f16_e32 v5, v255, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_mul_f16_e32 v5, v255, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_f16_e32 v5, v255, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_f16_e32 v5, v255, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_f16_e32 v5, v1, v255
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmaak_f16_e32 v5, v1, v255, 0xfe0b
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmac_f16_e32 v5, v1, v255
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmamk_f16_e32 v5, v1, 0xfe0b, v255
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_max_num_f16_e32 v5, v1, v255
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_min_num_f16_e32 v5, v1, v255
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_mul_f16_e32 v5, v1, v255
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_f16_e32 v5, v1, v255
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_f16_e32 v5, v1, v255
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_f16_dpp v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmac_f16_dpp v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_ldexp_f16_dpp v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_max_num_f16_dpp v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_min_num_f16_dpp v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_mul_f16_dpp v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_f16_dpp v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_f16_dpp v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_f16_dpp v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmac_f16_dpp v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_ldexp_f16_dpp v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_max_num_f16_dpp v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_min_num_f16_dpp v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_mul_f16_dpp v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_f16_dpp v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_f16_dpp v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_f16_dpp v5, v1, v255 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmac_f16_dpp v5, v1, v255 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_max_num_f16_dpp v5, v1, v255 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_min_num_f16_dpp v5, v1, v255 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_mul_f16_dpp v5, v1, v255 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_f16_dpp v5, v1, v255 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_f16_dpp v5, v1, v255 quad_perm:[3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_f16_dpp v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmac_f16_dpp v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_ldexp_f16_dpp v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_max_num_f16_dpp v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_min_num_f16_dpp v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_mul_f16_dpp v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_f16_dpp v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_f16_dpp v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_f16_dpp v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmac_f16_dpp v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_ldexp_f16_dpp v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_max_num_f16_dpp v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_min_num_f16_dpp v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_mul_f16_dpp v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_f16_dpp v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_f16_dpp v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_add_f16_dpp v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_fmac_f16_dpp v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_max_num_f16_dpp v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_min_num_f16_dpp v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_mul_f16_dpp v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_sub_f16_dpp v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_subrev_f16_dpp v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
