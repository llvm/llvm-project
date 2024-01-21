// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12 --implicit-check-not=_e32 %s

v_add_f16 v255, v1, v2
// GFX12: v_add_f16_e64

v_fmac_f16 v255, v1, v2
// GFX12: v_fmac_f16_e64

v_ldexp_f16 v255, v1, v2
// GFX12: v_ldexp_f16_e64

v_max_num_f16 v255, v1, v2
// GFX12: v_max_num_f16_e64

v_min_num_f16 v255, v1, v2
// GFX12: v_min_num_f16_e64

v_mul_f16 v255, v1, v2
// GFX12: v_mul_f16_e64

v_sub_f16 v255, v1, v2
// GFX12: v_sub_f16_e64

v_subrev_f16 v255, v1, v2
// GFX12: v_subrev_f16_e64

v_add_f16 v5, v255, v2
// GFX12: v_add_f16_e64

v_fmac_f16 v5, v255, v2
// GFX12: v_fmac_f16_e64

v_ldexp_f16 v5, v255, v2
// GFX12: v_ldexp_f16_e64

v_max_num_f16 v5, v255, v2
// GFX12: v_max_num_f16_e64

v_min_num_f16 v5, v255, v2
// GFX12: v_min_num_f16_e64

v_mul_f16 v5, v255, v2
// GFX12: v_mul_f16_e64

v_sub_f16 v5, v255, v2
// GFX12: v_sub_f16_e64

v_subrev_f16 v5, v255, v2
// GFX12: v_subrev_f16_e64

v_add_f16 v5, v1, v255
// GFX12: v_add_f16_e64

v_fmac_f16 v5, v1, v255
// GFX12: v_fmac_f16_e64

v_max_num_f16 v5, v1, v255
// GFX12: v_max_num_f16_e64

v_min_num_f16 v5, v1, v255
// GFX12: v_min_num_f16_e64

v_mul_f16 v5, v1, v255
// GFX12: v_mul_f16_e64

v_sub_f16 v5, v1, v255
// GFX12: v_sub_f16_e64

v_subrev_f16 v5, v1, v255
// GFX12: v_subrev_f16_e64

v_add_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: v_add_f16_e64

v_ldexp_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: v_ldexp_f16_e64

v_max_num_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: v_max_num_f16_e64

v_min_num_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: v_min_num_f16_e64

v_mul_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: v_mul_f16_e64

v_sub_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: v_sub_f16_e64

v_subrev_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX12: v_subrev_f16_e64

v_add_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: v_add_f16_e64

v_ldexp_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: v_ldexp_f16_e64

v_max_num_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: v_max_num_f16_e64

v_min_num_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: v_min_num_f16_e64

v_mul_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: v_mul_f16_e64

v_sub_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: v_sub_f16_e64

v_subrev_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX12: v_subrev_f16_e64

v_add_f16 v5, v1, v255 quad_perm:[3,2,1,0]
// GFX12: v_add_f16_e64

v_max_num_f16 v5, v1, v255 quad_perm:[3,2,1,0]
// GFX12: v_max_num_f16_e64

v_min_num_f16 v5, v1, v255 quad_perm:[3,2,1,0]
// GFX12: v_min_num_f16_e64

v_mul_f16 v5, v1, v255 quad_perm:[3,2,1,0]
// GFX12: v_mul_f16_e64

v_sub_f16 v5, v1, v255 quad_perm:[3,2,1,0]
// GFX12: v_sub_f16_e64

v_subrev_f16 v5, v1, v255 quad_perm:[3,2,1,0]
// GFX12: v_subrev_f16_e64

v_add_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_add_f16_e64

v_ldexp_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_ldexp_f16_e64

v_max_num_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_max_num_f16_e64

v_min_num_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_min_num_f16_e64

v_mul_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_mul_f16_e64

v_sub_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_sub_f16_e64

v_subrev_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_subrev_f16_e64

v_add_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_add_f16_e64

v_ldexp_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_ldexp_f16_e64

v_max_num_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_max_num_f16_e64

v_min_num_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_min_num_f16_e64

v_mul_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_mul_f16_e64

v_sub_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_sub_f16_e64

v_subrev_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_subrev_f16_e64

v_add_f16 v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_add_f16_e64

v_max_num_f16 v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_max_num_f16_e64

v_min_num_f16 v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_min_num_f16_e64

v_mul_f16 v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_mul_f16_e64

v_sub_f16 v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_sub_f16_e64

v_subrev_f16 v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX12: v_subrev_f16_e64
