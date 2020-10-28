// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=NOGFX10 --implicit-check-not=error: %s

// TODO: more helpful error message for missing dim operand
image_load v[0:3], v0, s[0:7] dmask:0xf unorm
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_load v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D da
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_load_pck v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D d16
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

image_load v[0:1], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image data size does not match dmask and tfe

image_load v[0:3], v[0:1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_load_mip v[0:3], v[0:2], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_CUBE
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_sample_d v[0:3], [v0, v1, v2, v3, v4], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D_ARRAY
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_sample_b_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_CUBE
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_sample_c_d v[0:3], [v0, v1, v2, v3, v4, v5, v6], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_sample_c_d_cl v[0:3], [v0, v1, v2, v3, v4, v5, v6, v7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_sample_c_d_cl_o v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: image address size does not match dim and a16

image_load v[0:1], v0, s[0:7] dmask:0x9 dim:1 D
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: failed parsing operand
