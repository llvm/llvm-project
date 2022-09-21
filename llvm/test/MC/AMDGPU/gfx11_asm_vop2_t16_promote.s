// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=_e32 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=_e32 %s

v_add_f16 v255, v1, v2
// GFX11: v_add_f16_e64

v_fmac_f16 v255, v1, v2
// GFX11: v_fmac_f16_e64

v_ldexp_f16 v255, v1, v2
// GFX11: v_ldexp_f16_e64

v_max_f16 v255, v1, v2
// GFX11: v_max_f16_e64

v_min_f16 v255, v1, v2
// GFX11: v_min_f16_e64

v_mul_f16 v255, v1, v2
// GFX11: v_mul_f16_e64

v_sub_f16 v255, v1, v2
// GFX11: v_sub_f16_e64

v_subrev_f16 v255, v1, v2
// GFX11: v_subrev_f16_e64

v_add_f16 v5, v255, v2
// GFX11: v_add_f16_e64

v_fmac_f16 v5, v255, v2
// GFX11: v_fmac_f16_e64

v_ldexp_f16 v5, v255, v2
// GFX11: v_ldexp_f16_e64

v_max_f16 v5, v255, v2
// GFX11: v_max_f16_e64

v_min_f16 v5, v255, v2
// GFX11: v_min_f16_e64

v_mul_f16 v5, v255, v2
// GFX11: v_mul_f16_e64

v_sub_f16 v5, v255, v2
// GFX11: v_sub_f16_e64

v_subrev_f16 v5, v255, v2
// GFX11: v_subrev_f16_e64

v_add_f16 v5, v1, v255
// GFX11: v_add_f16_e64

v_fmac_f16 v5, v1, v255
// GFX11: v_fmac_f16_e64

v_max_f16 v5, v1, v255
// GFX11: v_max_f16_e64

v_min_f16 v5, v1, v255
// GFX11: v_min_f16_e64

v_mul_f16 v5, v1, v255
// GFX11: v_mul_f16_e64

v_sub_f16 v5, v1, v255
// GFX11: v_sub_f16_e64

v_subrev_f16 v5, v1, v255
// GFX11: v_subrev_f16_e64

v_add_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_add_f16_e64

v_ldexp_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_ldexp_f16_e64

v_max_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_max_f16_e64

v_min_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_min_f16_e64

v_mul_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_mul_f16_e64

v_sub_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_sub_f16_e64

v_subrev_f16 v255, v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_subrev_f16_e64

v_add_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_add_f16_e64

v_ldexp_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_ldexp_f16_e64

v_max_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_max_f16_e64

v_min_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_min_f16_e64

v_mul_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_mul_f16_e64

v_sub_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_sub_f16_e64

v_subrev_f16 v5, v255, v2 quad_perm:[3,2,1,0]
// GFX11: v_subrev_f16_e64

v_add_f16 v5, v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_add_f16_e64

v_max_f16 v5, v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_max_f16_e64

v_min_f16 v5, v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_min_f16_e64

v_mul_f16 v5, v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_mul_f16_e64

v_sub_f16 v5, v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_sub_f16_e64

v_subrev_f16 v5, v1, v255 quad_perm:[3,2,1,0]
// GFX11: v_subrev_f16_e64

v_add_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_add_f16_e64

v_ldexp_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_ldexp_f16_e64

v_max_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_max_f16_e64

v_min_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_min_f16_e64

v_mul_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_mul_f16_e64

v_sub_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_sub_f16_e64

v_subrev_f16 v255, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_subrev_f16_e64

v_add_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_add_f16_e64

v_ldexp_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_ldexp_f16_e64

v_max_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_max_f16_e64

v_min_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_min_f16_e64

v_mul_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_mul_f16_e64

v_sub_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_sub_f16_e64

v_subrev_f16 v5, v255, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_subrev_f16_e64

v_add_f16 v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_add_f16_e64

v_max_f16 v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_max_f16_e64

v_min_f16 v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_min_f16_e64

v_mul_f16 v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_mul_f16_e64

v_sub_f16 v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_sub_f16_e64

v_subrev_f16 v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_subrev_f16_e64

