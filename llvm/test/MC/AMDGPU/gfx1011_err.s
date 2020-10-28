// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1011 %s 2>&1 | FileCheck --check-prefix=GFX10 --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1012 %s 2>&1 | FileCheck --check-prefix=GFX10 --implicit-check-not=error: %s

v_dot8c_i32_i4 v5, v1, v2
// GFX10: error: instruction not supported on this GPU

v_dot8c_i32_i4 v5, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// GFX10: error: not a valid operand.

v_dot8c_i32_i4 v5, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0 fi:1
// GFX10: error: not a valid operand.

v_dot8c_i32_i4 v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX10: error: not a valid operand.

v_dot8c_i32_i4 v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX10: error: not a valid operand.

s_getreg_b32 s2, hwreg(HW_REG_SHADER_CYCLES)
// GFX10: error: specified hardware register is not supported on this GPU

v_fma_legacy_f32 v0, v1, v2, v3
// GFX10: error: instruction not supported on this GPU

image_bvh_intersect_ray v[4:7], v[9:24], s[4:7]
// GFX10: error: instruction not supported on this GPU

image_bvh_intersect_ray v[4:7], v[9:16], s[4:7] a16
// GFX10: error: invalid operand

image_bvh64_intersect_ray v[4:7], v[9:24], s[4:7]
// GFX10: error: instruction not supported on this GPU

image_bvh64_intersect_ray v[4:7], v[9:24], s[4:7] a16
// GFX10: error: invalid operand

image_msaa_load v[1:4], v5, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_1D
// GFX10: error: not a valid operand.

image_msaa_load v[1:4], v5, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_1D glc
// GFX10: error: not a valid operand.

image_msaa_load v5, v[1:2], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D d16
// GFX10: error: not a valid operand.

image_msaa_load v[1:4], v5, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_1D
// GFX10: error: not a valid operand.

image_msaa_load v14, [v204,v11,v14,v19], s[40:47] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY
// GFX10: error: not a valid operand.
