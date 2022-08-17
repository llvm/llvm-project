// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX11,W32 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX11,W64 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=W32-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=W64-ERR --implicit-check-not=error: %s

v_cmp_le_u16_dpp v1, v2 dpp8:[7,7,7,3,4,4,6,7] fi:1
// GFX11: encoding: [0xea,0x04,0x76,0x7c,0x01,0xff,0x47,0xfa]

v_cmp_le_i16_dpp v1, v2 dpp8:[7,7,7,3,4,4,6,7]
// GFX11: encoding: [0xe9,0x04,0x66,0x7c,0x01,0xff,0x47,0xfa]

v_cmp_le_i32_dpp vcc_lo, v1, v255 dpp8:[0,2,1,3,4,5,6,7]
// W32: encoding: [0xe9,0xfe,0x87,0x7c,0x01,0x50,0xc6,0xfa]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error:

v_cmp_tru_f32_dpp vcc_lo, v1, v2 dpp8:[0,2,1,3,4,5,6,7]
// W32: encoding: [0xe9,0x04,0x3e,0x7c,0x01,0x50,0xc6,0xfa]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error:

// check vcc/vcc_lo have been added
v_cmp_lt_f32_dpp v1, v2 dpp8:[2,3,4,1,3,3,3,3]
// W32: v_cmp_lt_f32 vcc_lo, v1, v2 dpp8:[2,3,4,1,3,3,3,3] ; encoding: [0xe9,0x04,0x22,0x7c,0x01,0x1a,0xb3,0x6d]
// W64: v_cmp_lt_f32 vcc, v1, v2 dpp8:[2,3,4,1,3,3,3,3] ; encoding: [0xe9,0x04,0x22,0x7c,0x01,0x1a,0xb3,0x6d]

v_cmp_lt_u16_dpp vcc, v1, v2 dpp8:[7,6,5,3,4,2,1,0] fi:1
// W64: encoding: [0xea,0x04,0x72,0x7c,0x01,0x77,0x47,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error:

v_cmp_class_f16_dpp vcc, v1, v2 dpp8:[7,6,5,3,4,2,1,0] fi:1
// W64: encoding: [0xea,0x04,0xfa,0x7c,0x01,0x77,0x47,0x05]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error:
