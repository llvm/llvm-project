// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefixes=GFX11 %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefixes=GFX11 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefixes=W64-ERR  --implicit-check-not=error: %s

v_dual_mul_f32 v0, v0, v2 :: v_dual_mul_f32 v1, v1, v3
// GFX11: encoding: [0x00,0x05,0xc6,0xc8,0x01,0x07,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32      v0,     s1,     v2          ::  v_dual_mul_f32      v3,     s4,     v5
// GFX11: encoding: [0x01,0x04,0xc6,0xc8,0x04,0x0a,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32      v11,    v1,     v2          ::  v_dual_mul_f32      v10, 0x24681357, v5
// GFX11: encoding: [0x01,0x05,0xc6,0xc8,0xff,0x0a,0x0a,0x0b,0x57,0x13,0x68,0x24]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32      v11, 0x24681357, v2    ::  v_dual_mul_f32      v10, 0x24681357, v5
// GFX11: encoding: [0xff,0x04,0xc6,0xc8,0xff,0x0a,0x0a,0x0b,0x57,0x13,0x68,0x24]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_f32      v0,     v1 , v2                 ::  v_dual_max_f32      v3,     v4,     v5
// GFX11: encoding: [0x01,0x05,0xd4,0xca,0x04,0x0b,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32  v20,    v21,    v22 ::  v_dual_mov_b32      v41,    v42
// GFX11: encoding: [0x15,0x2d,0x50,0xca,0x2a,0x01,0x28,0x14]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32     v0,     v1,     v2          :: v_dual_fmamk_f32     v3, v6, 0x3f700000, v1
// GFX11: encoding: [0x01,0x05,0x04,0xc8,0x06,0x03,0x02,0x00,0x00,0x00,0x70,0x3f]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmamk_f32    v122, v74, 0xa0172923, v161 :: v_dual_lshlrev_b32 v247, v160, v99
// GFX11: encoding: [0x4a,0x43,0xa3,0xc8,0xa0,0xc7,0xf6,0x7a,0x23,0x29,0x17,0xa0]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmaak_f32    v122, s74, v161, 2.741 :: v_dual_and_b32 v247, v160, v98
// GFX11: encoding: [0x4a,0x42,0x65,0xc8,0xa0,0xc5,0xf6,0x7a,0x8b,0x6c,0x2f,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmaak_f32    v122, s74, v161, 2.741 :: v_dual_fmamk_f32     v3, v6, 2.741, v1
// GFX11: encoding: [0x4a,0x42,0x45,0xc8,0x06,0x03,0x02,0x7a,0x8b,0x6c,0x2f,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v247, v160 :: v_dual_fmaak_f32    v122, s74, v161, 2.741
// GFX11: encoding: [0xa0,0x01,0x02,0xca,0x4a,0x42,0x7b,0xf7,0x8b,0x6c,0x2f,0x40]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32      v0,     v1 , v2                 ::  v_dual_add_nc_u32      v3,     v4,     v5
// GFX11: encoding: [0x01,0x05,0xa0,0xc9,0x04,0x0b,0x02,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32      v11, 0x24681357, v2    ::  v_dual_dot2acc_f32_f16      v10, 0x24681357, v5
// GFX11: encoding: [0xff,0x04,0xd8,0xc9,0xff,0x0a,0x0a,0x0b,0x57,0x13,0x68,0x24]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 :: v_dual_fmamk_f32    v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX11: encoding: [0xff,0x42,0x85,0xc8,0xff,0x44,0x7b,0x7a,0xef,0xbe,0xad,0xde]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmamk_f32    v122, 255, 255, v161 :: v_dual_fmamk_f32    v123, 255, 255, v162
// GFX11: encoding: [0xff,0x42,0x85,0xc8,0xff,0x44,0x7b,0x7a,0xff,0x00,0x00,0x00]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

;Illegal, but assembler does not check register or literal constraints for VOPD
;v_dual_fmamk_f32    v122, v74, 0xdeadbeef, v161 :: v_dual_fmamk_f32    v122, v74, 0xa0172923, v161
