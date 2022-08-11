// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX11,W32 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX11,W64 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=W32-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=W64-ERR --implicit-check-not=error: %s

v_cmp_gt_u16_dpp v1, v2 row_shl:0x7 row_mask:0x0 bank_mask:0x0 fi:1
// GFX11: encoding: [0xfa,0x04,0x78,0x7c,0x01,0x07,0x05,0x00]

v_cmp_gt_i16_dpp v1, v2 quad_perm:[1,3,1,0] row_mask:0x7
// GFX11: encoding: [0xfa,0x04,0x68,0x7c,0x01,0x1d,0x00,0x7f]

v_cmp_lt_f32 v1, -v2 quad_perm:[0,1,2,2]
// GFX11: encoding: [0xfa,0x04,0x22,0x7c,0x01,0xa4,0x40,0xff]

v_cmp_gt_i32_dpp vcc_lo, v1, v255 row_mirror bank_mask:0x2 fi:1
// W32: encoding: [0xfa,0xfe,0x89,0x7c,0x01,0x40,0x05,0xf2]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error:

v_cmp_f_f32_dpp vcc_lo, v1, v2 row_shl:0x7 row_mask:0x0 bank_mask:0x0
// W32: encoding: [0xfa,0x04,0x20,0x7c,0x01,0x07,0x01,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error:

v_cmp_ge_u16_dpp vcc, v1, v2 row_share:0xa bound_ctrl:0
// W64: encoding: [0xfa,0x04,0x7c,0x7c,0x01,0x5a,0x09,0xff]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error:

v_cmp_class_f16_dpp vcc, v1, v2 row_half_mirror bound_ctrl:0
// W64: encoding: [0xfa,0x04,0xfa,0x7c,0x01,0x41,0x09,0xff]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error:
