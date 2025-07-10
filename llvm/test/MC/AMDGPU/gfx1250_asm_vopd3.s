// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s | FileCheck --check-prefixes=GFX1250 %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s | FileCheck --check-prefixes=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefixes=W64-ERR --implicit-check-not=error: %s

v_dual_add_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3 ; encoding: [0x04,0x41,0x10,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3 ; encoding: [0x01,0x41,0x10,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3 ; encoding: [0xff,0x41,0x10,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3 ; encoding: [0x02,0x41,0x10,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3 ; encoding: [0x03,0x41,0x10,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3 ; encoding: [0x69,0x40,0x10,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3 ; encoding: [0x01,0x40,0x10,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x10,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x10,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x10,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3 ; encoding: [0x7d,0x40,0x10,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x10,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x10,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3 ; encoding: [0xfd,0x40,0x10,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x10,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x10,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3 ; encoding: [0x04,0x01,0x11,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3 ; encoding: [0x01,0x01,0x11,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3 ; encoding: [0xff,0x01,0x11,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3 ; encoding: [0x02,0x01,0x11,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3 ; encoding: [0x03,0x01,0x11,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3 ; encoding: [0x69,0x00,0x11,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3 ; encoding: [0x01,0x00,0x11,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x11,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x11,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x11,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x00,0x11,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x11,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x11,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x00,0x11,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x11,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x11,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo ; encoding: [0x04,0x91,0x10,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo ; encoding: [0x01,0x91,0x10,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo ; encoding: [0xff,0x91,0x10,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo ; encoding: [0x02,0x91,0x10,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo ; encoding: [0x03,0x91,0x10,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo ; encoding: [0x69,0x90,0x10,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo ; encoding: [0x01,0x90,0x10,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo ; encoding: [0x7b,0x90,0x10,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo ; encoding: [0x7f,0x90,0x10,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo ; encoding: [0x7e,0x90,0x10,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo ; encoding: [0x7d,0x90,0x10,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo ; encoding: [0x6b,0x90,0x10,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo ; encoding: [0x6a,0x90,0x10,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo ; encoding: [0xfd,0x90,0x10,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo ; encoding: [0xf0,0x90,0x10,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo ; encoding: [0xc1,0x90,0x10,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3 ; encoding: [0x04,0x01,0x10,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3 ; encoding: [0x01,0x01,0x10,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3 ; encoding: [0xff,0x01,0x10,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3 ; encoding: [0x02,0x01,0x10,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3 ; encoding: [0x03,0x01,0x10,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3 ; encoding: [0x69,0x00,0x10,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3 ; encoding: [0x01,0x00,0x10,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x10,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x10,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x10,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3 ; encoding: [0x7d,0x00,0x10,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x10,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x10,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3 ; encoding: [0xfd,0x00,0x10,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x10,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x10,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3 ; encoding: [0x04,0x11,0x11,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3 ; encoding: [0x01,0x11,0x11,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3 ; encoding: [0xff,0x11,0x11,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3 ; encoding: [0x02,0x11,0x11,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3 ; encoding: [0x03,0x11,0x11,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3 ; encoding: [0x69,0x10,0x11,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3 ; encoding: [0x01,0x10,0x11,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3 ; encoding: [0x7b,0x10,0x11,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3 ; encoding: [0x7f,0x10,0x11,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3 ; encoding: [0x7e,0x10,0x11,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3 ; encoding: [0x7d,0x10,0x11,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3 ; encoding: [0x6b,0x10,0x11,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3 ; encoding: [0x6a,0x10,0x11,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3 ; encoding: [0xfd,0x10,0x11,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2 ; encoding: [0xf0,0x10,0x11,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5 ; encoding: [0xc1,0x10,0x11,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3 ; encoding: [0x04,0xa1,0x10,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3 ; encoding: [0x01,0xa1,0x10,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3 ; encoding: [0xff,0xa1,0x10,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3 ; encoding: [0x02,0xa1,0x10,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3 ; encoding: [0x03,0xa1,0x10,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3 ; encoding: [0x69,0xa0,0x10,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3 ; encoding: [0x01,0xa0,0x10,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xa0,0x10,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xa0,0x10,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xa0,0x10,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3 ; encoding: [0x7d,0xa0,0x10,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xa0,0x10,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xa0,0x10,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3 ; encoding: [0xfd,0xa0,0x10,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xa0,0x10,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xa0,0x10,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3 ; encoding: [0x04,0xb1,0x10,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3 ; encoding: [0x01,0xb1,0x10,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3 ; encoding: [0xff,0xb1,0x10,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3 ; encoding: [0x02,0xb1,0x10,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3 ; encoding: [0x03,0xb1,0x10,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3 ; encoding: [0x69,0xb0,0x10,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3 ; encoding: [0x01,0xb0,0x10,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xb0,0x10,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xb0,0x10,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xb0,0x10,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3 ; encoding: [0x7d,0xb0,0x10,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xb0,0x10,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xb0,0x10,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3 ; encoding: [0xfd,0xb0,0x10,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xb0,0x10,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xb0,0x10,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1
// GFX1250: v_dual_add_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1 ; encoding: [0x04,0x81,0x10,0xcf,0x01,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255
// GFX1250: v_dual_add_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255 ; encoding: [0x01,0x81,0x10,0xcf,0xff,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2
// GFX1250: v_dual_add_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2 ; encoding: [0xff,0x81,0x10,0xcf,0x02,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3
// GFX1250: v_dual_add_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3 ; encoding: [0x02,0x81,0x10,0xcf,0x03,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4
// GFX1250: v_dual_add_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4 ; encoding: [0x03,0x81,0x10,0xcf,0x04,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1
// GFX1250: v_dual_add_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1 ; encoding: [0x69,0x80,0x10,0xcf,0x01,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105
// GFX1250: v_dual_add_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105 ; encoding: [0x01,0x80,0x10,0xcf,0x69,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo
// GFX1250: v_dual_add_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo ; encoding: [0x7b,0x80,0x10,0xcf,0x6a,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi
// GFX1250: v_dual_add_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi ; encoding: [0x7f,0x80,0x10,0xcf,0x6b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15
// GFX1250: v_dual_add_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15 ; encoding: [0x7e,0x80,0x10,0xcf,0x7b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0
// GFX1250: v_dual_add_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0 ; encoding: [0x7d,0x80,0x10,0xcf,0x7d,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo
// GFX1250: v_dual_add_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo ; encoding: [0x6b,0x80,0x10,0xcf,0x7e,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi
// GFX1250: v_dual_add_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi ; encoding: [0x6a,0x80,0x10,0xcf,0x7f,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1
// GFX1250: v_dual_add_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1 ; encoding: [0xfd,0x80,0x10,0xcf,0xc1,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5 ; encoding: [0xf0,0x80,0x10,0xcf,0xf0,0x00,0x03,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc ; encoding: [0xc1,0x80,0x10,0xcf,0xfd,0x00,0x04,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3 ; encoding: [0x04,0x71,0x10,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3 ; encoding: [0x01,0x71,0x10,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3 ; encoding: [0xff,0x71,0x10,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3 ; encoding: [0x02,0x71,0x10,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3 ; encoding: [0x03,0x71,0x10,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3 ; encoding: [0x69,0x70,0x10,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3 ; encoding: [0x01,0x70,0x10,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x10,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x10,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x10,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3 ; encoding: [0x7d,0x70,0x10,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x10,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x10,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3 ; encoding: [0xfd,0x70,0x10,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x10,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x10,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3 ; encoding: [0x04,0x31,0x10,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3 ; encoding: [0x01,0x31,0x10,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3 ; encoding: [0xff,0x31,0x10,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3 ; encoding: [0x02,0x31,0x10,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3 ; encoding: [0x03,0x31,0x10,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3 ; encoding: [0x69,0x30,0x10,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3 ; encoding: [0x01,0x30,0x10,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x30,0x10,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x30,0x10,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x30,0x10,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3 ; encoding: [0x7d,0x30,0x10,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x30,0x10,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x30,0x10,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3 ; encoding: [0xfd,0x30,0x10,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2 ; encoding: [0xf0,0x30,0x10,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5 ; encoding: [0xc1,0x30,0x10,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3 ; encoding: [0x04,0x51,0x10,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3 ; encoding: [0x01,0x51,0x10,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3 ; encoding: [0xff,0x51,0x10,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3 ; encoding: [0x02,0x51,0x10,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3 ; encoding: [0x03,0x51,0x10,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3 ; encoding: [0x69,0x50,0x10,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3 ; encoding: [0x01,0x50,0x10,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x50,0x10,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x50,0x10,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x50,0x10,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3 ; encoding: [0x7d,0x50,0x10,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x50,0x10,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x50,0x10,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3 ; encoding: [0xfd,0x50,0x10,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2 ; encoding: [0xf0,0x50,0x10,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5 ; encoding: [0xc1,0x50,0x10,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3 ; encoding: [0x04,0x61,0x10,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3 ; encoding: [0x01,0x61,0x10,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3 ; encoding: [0xff,0x61,0x10,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3 ; encoding: [0x02,0x61,0x10,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3 ; encoding: [0x03,0x61,0x10,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3 ; encoding: [0x69,0x60,0x10,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3 ; encoding: [0x01,0x60,0x10,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x60,0x10,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x60,0x10,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x60,0x10,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3 ; encoding: [0x7d,0x60,0x10,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x60,0x10,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x60,0x10,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3 ; encoding: [0xfd,0x60,0x10,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2 ; encoding: [0xf0,0x60,0x10,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5 ; encoding: [0xc1,0x60,0x10,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4 ; encoding: [0x04,0x31,0x11,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x04,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 ; encoding: [0x04,0x21,0x11,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_add_f32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_add_f32 v7, v1, v3 ; encoding: [0x04,0x41,0x24,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_add_f32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_add_f32 v7, v255, v3 ; encoding: [0x01,0x41,0x24,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_add_f32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_add_f32 v7, v2, v3 ; encoding: [0xff,0x41,0x24,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_add_f32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_add_f32 v7, v3, v3 ; encoding: [0x02,0x41,0x24,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_add_f32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_add_f32 v7, v4, v3 ; encoding: [0x03,0x41,0x24,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_add_f32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_add_f32 v7, s105, v3 ; encoding: [0x69,0x40,0x24,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_add_f32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_add_f32 v7, s1, v3 ; encoding: [0x01,0x40,0x24,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_add_f32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_add_f32 v7, ttmp15, v3 ; encoding: [0x7b,0x40,0x24,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_add_f32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_add_f32 v7, exec_hi, v3 ; encoding: [0x7f,0x40,0x24,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_add_f32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_add_f32 v7, exec_lo, v3 ; encoding: [0x7e,0x40,0x24,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_add_f32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_add_f32 v7, m0, v3 ; encoding: [0x7d,0x40,0x24,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_add_f32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_add_f32 v7, vcc_hi, v3 ; encoding: [0x6b,0x40,0x24,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_add_f32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_add_f32 v7, vcc_lo, v3 ; encoding: [0x6a,0x40,0x24,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_add_f32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_add_f32 v7, -1, v3 ; encoding: [0xfd,0x40,0x24,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_add_f32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_add_f32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x24,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_add_f32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_add_f32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x24,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_add_nc_u32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_add_nc_u32 v7, v1, v3 ; encoding: [0x04,0x01,0x25,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_add_nc_u32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_add_nc_u32 v7, v255, v3 ; encoding: [0x01,0x01,0x25,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_add_nc_u32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_add_nc_u32 v7, v2, v3 ; encoding: [0xff,0x01,0x25,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_add_nc_u32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_add_nc_u32 v7, v3, v3 ; encoding: [0x02,0x01,0x25,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_add_nc_u32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_add_nc_u32 v7, v4, v3 ; encoding: [0x03,0x01,0x25,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_add_nc_u32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_add_nc_u32 v7, s105, v3 ; encoding: [0x69,0x00,0x25,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_add_nc_u32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_add_nc_u32 v7, s1, v3 ; encoding: [0x01,0x00,0x25,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_add_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_add_nc_u32 v7, ttmp15, v3 ; encoding: [0x7b,0x00,0x25,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_add_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_add_nc_u32 v7, exec_hi, v3 ; encoding: [0x7f,0x00,0x25,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_add_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_add_nc_u32 v7, exec_lo, v3 ; encoding: [0x7e,0x00,0x25,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_add_nc_u32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_add_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x00,0x25,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_add_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_add_nc_u32 v7, vcc_hi, v3 ; encoding: [0x6b,0x00,0x25,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_add_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_add_nc_u32 v7, vcc_lo, v3 ; encoding: [0x6a,0x00,0x25,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_add_nc_u32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_add_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x00,0x25,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_add_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_add_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x25,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_add_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_add_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x25,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo ; encoding: [0x04,0x91,0x24,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo ; encoding: [0x01,0x91,0x24,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo ; encoding: [0xff,0x91,0x24,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo ; encoding: [0x02,0x91,0x24,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo ; encoding: [0x03,0x91,0x24,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo ; encoding: [0x69,0x90,0x24,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo ; encoding: [0x01,0x90,0x24,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo ; encoding: [0x7b,0x90,0x24,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo ; encoding: [0x7f,0x90,0x24,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo ; encoding: [0x7e,0x90,0x24,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo ; encoding: [0x7d,0x90,0x24,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo ; encoding: [0x6b,0x90,0x24,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo ; encoding: [0x6a,0x90,0x24,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo ; encoding: [0xfd,0x90,0x24,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo ; encoding: [0xf0,0x90,0x24,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo ; encoding: [0xc1,0x90,0x24,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v28, -v15, v15, s46 :: v_dual_cndmask_b32 v29, -v13, -v13, s46
// GFX1250: v_dual_cndmask_b32 v28, -v15, v15, s46 :: v_dual_cndmask_b32 v29, -v13, -v13, s46 ; encoding: [0x0f,0x91,0x24,0xcf,0x0d,0x33,0x0f,0x2e,0x1c,0x0d,0x2e,0x1d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_fmac_f32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_fmac_f32 v7, v1, v3 ; encoding: [0x04,0x01,0x24,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_fmac_f32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_fmac_f32 v7, v255, v3 ; encoding: [0x01,0x01,0x24,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_fmac_f32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_fmac_f32 v7, v2, v3 ; encoding: [0xff,0x01,0x24,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_fmac_f32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_fmac_f32 v7, v3, v3 ; encoding: [0x02,0x01,0x24,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_fmac_f32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_fmac_f32 v7, v4, v3 ; encoding: [0x03,0x01,0x24,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_fmac_f32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_fmac_f32 v7, s105, v3 ; encoding: [0x69,0x00,0x24,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_fmac_f32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_fmac_f32 v7, s1, v3 ; encoding: [0x01,0x00,0x24,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_fmac_f32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_fmac_f32 v7, ttmp15, v3 ; encoding: [0x7b,0x00,0x24,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_fmac_f32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_fmac_f32 v7, exec_hi, v3 ; encoding: [0x7f,0x00,0x24,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_fmac_f32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_fmac_f32 v7, exec_lo, v3 ; encoding: [0x7e,0x00,0x24,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_fmac_f32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_fmac_f32 v7, m0, v3 ; encoding: [0x7d,0x00,0x24,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_fmac_f32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_fmac_f32 v7, vcc_hi, v3 ; encoding: [0x6b,0x00,0x24,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_fmac_f32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_fmac_f32 v7, vcc_lo, v3 ; encoding: [0x6a,0x00,0x24,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_fmac_f32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_fmac_f32 v7, -1, v3 ; encoding: [0xfd,0x00,0x24,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_fmac_f32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_fmac_f32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x24,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_fmac_f32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_fmac_f32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x24,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_lshlrev_b32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_lshlrev_b32 v7, v1, v3 ; encoding: [0x04,0x11,0x25,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_lshlrev_b32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_lshlrev_b32 v7, v255, v3 ; encoding: [0x01,0x11,0x25,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_lshlrev_b32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_lshlrev_b32 v7, v2, v3 ; encoding: [0xff,0x11,0x25,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_lshlrev_b32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_lshlrev_b32 v7, v3, v3 ; encoding: [0x02,0x11,0x25,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_lshlrev_b32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_lshlrev_b32 v7, v4, v3 ; encoding: [0x03,0x11,0x25,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_lshlrev_b32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_lshlrev_b32 v7, s105, v3 ; encoding: [0x69,0x10,0x25,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_lshlrev_b32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_lshlrev_b32 v7, s1, v3 ; encoding: [0x01,0x10,0x25,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_lshlrev_b32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_lshlrev_b32 v7, ttmp15, v3 ; encoding: [0x7b,0x10,0x25,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_lshlrev_b32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_lshlrev_b32 v7, exec_hi, v3 ; encoding: [0x7f,0x10,0x25,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_lshlrev_b32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_lshlrev_b32 v7, exec_lo, v3 ; encoding: [0x7e,0x10,0x25,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_lshlrev_b32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_lshlrev_b32 v7, m0, v3 ; encoding: [0x7d,0x10,0x25,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_lshlrev_b32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_lshlrev_b32 v7, vcc_hi, v3 ; encoding: [0x6b,0x10,0x25,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_lshlrev_b32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_lshlrev_b32 v7, vcc_lo, v3 ; encoding: [0x6a,0x10,0x25,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_lshlrev_b32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_lshlrev_b32 v7, -1, v3 ; encoding: [0xfd,0x10,0x25,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_lshlrev_b32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_lshlrev_b32 v7, 0.5, v2 ; encoding: [0xf0,0x10,0x25,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_lshlrev_b32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_lshlrev_b32 v7, src_scc, v5 ; encoding: [0xc1,0x10,0x25,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_max_num_f32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_max_num_f32 v7, v1, v3 ; encoding: [0x04,0xa1,0x24,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_max_num_f32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_max_num_f32 v7, v255, v3 ; encoding: [0x01,0xa1,0x24,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_max_num_f32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_max_num_f32 v7, v2, v3 ; encoding: [0xff,0xa1,0x24,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_max_num_f32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_max_num_f32 v7, v3, v3 ; encoding: [0x02,0xa1,0x24,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_max_num_f32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_max_num_f32 v7, v4, v3 ; encoding: [0x03,0xa1,0x24,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_max_num_f32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_max_num_f32 v7, s105, v3 ; encoding: [0x69,0xa0,0x24,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_max_num_f32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_max_num_f32 v7, s1, v3 ; encoding: [0x01,0xa0,0x24,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_max_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_max_num_f32 v7, ttmp15, v3 ; encoding: [0x7b,0xa0,0x24,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_max_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_max_num_f32 v7, exec_hi, v3 ; encoding: [0x7f,0xa0,0x24,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_max_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_max_num_f32 v7, exec_lo, v3 ; encoding: [0x7e,0xa0,0x24,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_max_num_f32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_max_num_f32 v7, m0, v3 ; encoding: [0x7d,0xa0,0x24,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_max_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_max_num_f32 v7, vcc_hi, v3 ; encoding: [0x6b,0xa0,0x24,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_max_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_max_num_f32 v7, vcc_lo, v3 ; encoding: [0x6a,0xa0,0x24,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_max_num_f32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_max_num_f32 v7, -1, v3 ; encoding: [0xfd,0xa0,0x24,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_max_num_f32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_max_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xa0,0x24,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_max_num_f32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_max_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xa0,0x24,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_min_num_f32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_min_num_f32 v7, v1, v3 ; encoding: [0x04,0xb1,0x24,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_min_num_f32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_min_num_f32 v7, v255, v3 ; encoding: [0x01,0xb1,0x24,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_min_num_f32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_min_num_f32 v7, v2, v3 ; encoding: [0xff,0xb1,0x24,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_min_num_f32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_min_num_f32 v7, v3, v3 ; encoding: [0x02,0xb1,0x24,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_min_num_f32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_min_num_f32 v7, v4, v3 ; encoding: [0x03,0xb1,0x24,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_min_num_f32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_min_num_f32 v7, s105, v3 ; encoding: [0x69,0xb0,0x24,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_min_num_f32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_min_num_f32 v7, s1, v3 ; encoding: [0x01,0xb0,0x24,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_min_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_min_num_f32 v7, ttmp15, v3 ; encoding: [0x7b,0xb0,0x24,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_min_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_min_num_f32 v7, exec_hi, v3 ; encoding: [0x7f,0xb0,0x24,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_min_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_min_num_f32 v7, exec_lo, v3 ; encoding: [0x7e,0xb0,0x24,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_min_num_f32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_min_num_f32 v7, m0, v3 ; encoding: [0x7d,0xb0,0x24,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_min_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_min_num_f32 v7, vcc_hi, v3 ; encoding: [0x6b,0xb0,0x24,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_min_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_min_num_f32 v7, vcc_lo, v3 ; encoding: [0x6a,0xb0,0x24,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_min_num_f32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_min_num_f32 v7, -1, v3 ; encoding: [0xfd,0xb0,0x24,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_min_num_f32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_min_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xb0,0x24,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_min_num_f32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_min_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xb0,0x24,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v255, vcc_lo :: v_dual_mov_b32 v7, v1
// GFX1250: v_dual_cndmask_b32 v255, v4, v255, vcc_lo :: v_dual_mov_b32 v7, v1 ; encoding: [0x04,0x81,0x24,0xcf,0x01,0x01,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v255, vcc_lo :: v_dual_mov_b32 v7, v255
// GFX1250: v_dual_cndmask_b32 v255, v1, v255, vcc_lo :: v_dual_mov_b32 v7, v255 ; encoding: [0x01,0x81,0x24,0xcf,0xff,0x01,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v255, vcc_lo :: v_dual_mov_b32 v7, v2
// GFX1250: v_dual_cndmask_b32 v255, v255, v255, vcc_lo :: v_dual_mov_b32 v7, v2 ; encoding: [0xff,0x81,0x24,0xcf,0x02,0x01,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v255, vcc_lo :: v_dual_mov_b32 v7, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v255, vcc_lo :: v_dual_mov_b32 v7, v3 ; encoding: [0x02,0x81,0x24,0xcf,0x03,0x01,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v255, vcc_lo :: v_dual_mov_b32 v7, v4
// GFX1250: v_dual_cndmask_b32 v255, v3, v255, vcc_lo :: v_dual_mov_b32 v7, v4 ; encoding: [0x03,0x81,0x24,0xcf,0x04,0x01,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v255, vcc_lo :: v_dual_mov_b32 v7, s105
// GFX1250: v_dual_cndmask_b32 v255, s105, v255, vcc_lo :: v_dual_mov_b32 v7, s105 ; encoding: [0x69,0x80,0x24,0xcf,0x69,0x00,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v255, vcc_lo :: v_dual_mov_b32 v7, s1
// GFX1250: v_dual_cndmask_b32 v255, s1, v255, vcc_lo :: v_dual_mov_b32 v7, s1 ; encoding: [0x01,0x80,0x24,0xcf,0x01,0x00,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v255, vcc_lo :: v_dual_mov_b32 v7, ttmp15
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v255, vcc_lo :: v_dual_mov_b32 v7, ttmp15 ; encoding: [0x7b,0x80,0x24,0xcf,0x7b,0x00,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v255, vcc_lo :: v_dual_mov_b32 v7, exec_hi
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v255, vcc_lo :: v_dual_mov_b32 v7, exec_hi ; encoding: [0x7f,0x80,0x24,0xcf,0x7f,0x00,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v255, vcc_lo :: v_dual_mov_b32 v7, exec_lo
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v255, vcc_lo :: v_dual_mov_b32 v7, exec_lo ; encoding: [0x7e,0x80,0x24,0xcf,0x7e,0x00,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v255, vcc_lo :: v_dual_mov_b32 v7, m0
// GFX1250: v_dual_cndmask_b32 v255, m0, v255, vcc_lo :: v_dual_mov_b32 v7, m0 ; encoding: [0x7d,0x80,0x24,0xcf,0x7d,0x00,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v255, vcc_lo :: v_dual_mov_b32 v7, vcc_hi
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v255, vcc_lo :: v_dual_mov_b32 v7, vcc_hi ; encoding: [0x6b,0x80,0x24,0xcf,0x6b,0x00,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v255, vcc_lo :: v_dual_mov_b32 v7, vcc_lo
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v255, vcc_lo :: v_dual_mov_b32 v7, vcc_lo ; encoding: [0x6a,0x80,0x24,0xcf,0x6a,0x00,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v255, vcc_lo :: v_dual_mov_b32 v7, -1
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v255, vcc_lo :: v_dual_mov_b32 v7, -1 ; encoding: [0xfd,0x80,0x24,0xcf,0xc1,0x00,0xff,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_mov_b32 v7, 0.5
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_mov_b32 v7, 0.5 ; encoding: [0xf0,0x80,0x24,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_mov_b32 v7, src_scc
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_mov_b32 v7, src_scc ; encoding: [0xc1,0x80,0x24,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, v1, v3 ; encoding: [0x04,0x71,0x24,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, v255, v3 ; encoding: [0x01,0x71,0x24,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, v2, v3 ; encoding: [0xff,0x71,0x24,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, v3, v3 ; encoding: [0x02,0x71,0x24,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, v4, v3 ; encoding: [0x03,0x71,0x24,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, s105, v3 ; encoding: [0x69,0x70,0x24,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, s1, v3 ; encoding: [0x01,0x70,0x24,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3 ; encoding: [0x7b,0x70,0x24,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3 ; encoding: [0x7f,0x70,0x24,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3 ; encoding: [0x7e,0x70,0x24,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, m0, v3 ; encoding: [0x7d,0x70,0x24,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3 ; encoding: [0x6b,0x70,0x24,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3 ; encoding: [0x6a,0x70,0x24,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, -1, v3 ; encoding: [0xfd,0x70,0x24,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x24,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x24,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_mul_f32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_mul_f32 v7, v1, v3 ; encoding: [0x04,0x31,0x24,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_mul_f32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_mul_f32 v7, v255, v3 ; encoding: [0x01,0x31,0x24,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_mul_f32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_mul_f32 v7, v2, v3 ; encoding: [0xff,0x31,0x24,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_mul_f32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_mul_f32 v7, v3, v3 ; encoding: [0x02,0x31,0x24,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_mul_f32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_mul_f32 v7, v4, v3 ; encoding: [0x03,0x31,0x24,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_mul_f32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_mul_f32 v7, s105, v3 ; encoding: [0x69,0x30,0x24,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_mul_f32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_mul_f32 v7, s1, v3 ; encoding: [0x01,0x30,0x24,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_mul_f32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_mul_f32 v7, ttmp15, v3 ; encoding: [0x7b,0x30,0x24,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_mul_f32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_mul_f32 v7, exec_hi, v3 ; encoding: [0x7f,0x30,0x24,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_mul_f32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_mul_f32 v7, exec_lo, v3 ; encoding: [0x7e,0x30,0x24,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_mul_f32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_mul_f32 v7, m0, v3 ; encoding: [0x7d,0x30,0x24,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_mul_f32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_mul_f32 v7, vcc_hi, v3 ; encoding: [0x6b,0x30,0x24,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_mul_f32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_mul_f32 v7, vcc_lo, v3 ; encoding: [0x6a,0x30,0x24,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_mul_f32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_mul_f32 v7, -1, v3 ; encoding: [0xfd,0x30,0x24,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_mul_f32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_mul_f32 v7, 0.5, v2 ; encoding: [0xf0,0x30,0x24,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_mul_f32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_mul_f32 v7, src_scc, v5 ; encoding: [0xc1,0x30,0x24,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_sub_f32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_sub_f32 v7, v1, v3 ; encoding: [0x04,0x51,0x24,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_sub_f32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_sub_f32 v7, v255, v3 ; encoding: [0x01,0x51,0x24,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_sub_f32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_sub_f32 v7, v2, v3 ; encoding: [0xff,0x51,0x24,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_sub_f32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_sub_f32 v7, v3, v3 ; encoding: [0x02,0x51,0x24,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_sub_f32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_sub_f32 v7, v4, v3 ; encoding: [0x03,0x51,0x24,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_sub_f32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_sub_f32 v7, s105, v3 ; encoding: [0x69,0x50,0x24,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_sub_f32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_sub_f32 v7, s1, v3 ; encoding: [0x01,0x50,0x24,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_sub_f32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_sub_f32 v7, ttmp15, v3 ; encoding: [0x7b,0x50,0x24,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_sub_f32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_sub_f32 v7, exec_hi, v3 ; encoding: [0x7f,0x50,0x24,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_sub_f32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_sub_f32 v7, exec_lo, v3 ; encoding: [0x7e,0x50,0x24,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_sub_f32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_sub_f32 v7, m0, v3 ; encoding: [0x7d,0x50,0x24,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_sub_f32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_sub_f32 v7, vcc_hi, v3 ; encoding: [0x6b,0x50,0x24,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_sub_f32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_sub_f32 v7, vcc_lo, v3 ; encoding: [0x6a,0x50,0x24,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_sub_f32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_sub_f32 v7, -1, v3 ; encoding: [0xfd,0x50,0x24,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_sub_f32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_sub_f32 v7, 0.5, v2 ; encoding: [0xf0,0x50,0x24,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_sub_f32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_sub_f32 v7, src_scc, v5 ; encoding: [0xc1,0x50,0x24,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_subrev_f32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_subrev_f32 v7, v1, v3 ; encoding: [0x04,0x61,0x24,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_subrev_f32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_subrev_f32 v7, v255, v3 ; encoding: [0x01,0x61,0x24,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_subrev_f32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_subrev_f32 v7, v2, v3 ; encoding: [0xff,0x61,0x24,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_subrev_f32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_subrev_f32 v7, v3, v3 ; encoding: [0x02,0x61,0x24,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_subrev_f32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_subrev_f32 v7, v4, v3 ; encoding: [0x03,0x61,0x24,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_subrev_f32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_subrev_f32 v7, s105, v3 ; encoding: [0x69,0x60,0x24,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_subrev_f32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_subrev_f32 v7, s1, v3 ; encoding: [0x01,0x60,0x24,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_subrev_f32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_subrev_f32 v7, ttmp15, v3 ; encoding: [0x7b,0x60,0x24,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_subrev_f32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_subrev_f32 v7, exec_hi, v3 ; encoding: [0x7f,0x60,0x24,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_subrev_f32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_subrev_f32 v7, exec_lo, v3 ; encoding: [0x7e,0x60,0x24,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_subrev_f32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_subrev_f32 v7, m0, v3 ; encoding: [0x7d,0x60,0x24,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_subrev_f32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_subrev_f32 v7, vcc_hi, v3 ; encoding: [0x6b,0x60,0x24,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_subrev_f32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_subrev_f32 v7, vcc_lo, v3 ; encoding: [0x6a,0x60,0x24,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_subrev_f32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_subrev_f32 v7, -1, v3 ; encoding: [0xfd,0x60,0x24,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_subrev_f32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_subrev_f32 v7, 0.5, v2 ; encoding: [0xf0,0x60,0x24,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_subrev_f32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_subrev_f32 v7, src_scc, v5 ; encoding: [0xc1,0x60,0x24,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_fma_f32 v7, v1, v3, v4
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_fma_f32 v7, v1, v3, v4 ; encoding: [0x04,0x31,0x25,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x04,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_bitop2_b32 v7, v1, v3 bitop3:1
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_bitop2_b32 v7, v1, v3 bitop3:1 ; encoding: [0x04,0x21,0x25,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x01,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3 ; encoding: [0x04,0x41,0x00,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3 ; encoding: [0x01,0x41,0x00,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3 ; encoding: [0xff,0x41,0x00,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3 ; encoding: [0x02,0x41,0x00,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3 ; encoding: [0x03,0x41,0x00,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3 ; encoding: [0x69,0x40,0x00,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3 ; encoding: [0x01,0x40,0x00,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x00,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x00,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x00,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3 ; encoding: [0x7d,0x40,0x00,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x00,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x00,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3 ; encoding: [0xfd,0x40,0x00,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x00,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x00,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3 ; encoding: [0x04,0x01,0x01,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3 ; encoding: [0x01,0x01,0x01,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3 ; encoding: [0xff,0x01,0x01,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3 ; encoding: [0x02,0x01,0x01,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3 ; encoding: [0x03,0x01,0x01,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3 ; encoding: [0x69,0x00,0x01,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3 ; encoding: [0x01,0x00,0x01,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x01,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x01,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x01,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x00,0x01,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x01,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x01,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x00,0x01,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x01,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x01,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo ; encoding: [0x04,0x91,0x00,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo ; encoding: [0x01,0x91,0x00,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo ; encoding: [0xff,0x91,0x00,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo ; encoding: [0x02,0x91,0x00,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo ; encoding: [0x03,0x91,0x00,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo ; encoding: [0x69,0x90,0x00,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo ; encoding: [0x01,0x90,0x00,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo ; encoding: [0x7b,0x90,0x00,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo ; encoding: [0x7f,0x90,0x00,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo ; encoding: [0x7e,0x90,0x00,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo ; encoding: [0x7d,0x90,0x00,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo ; encoding: [0x6b,0x90,0x00,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo ; encoding: [0x6a,0x90,0x00,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo ; encoding: [0xfd,0x90,0x00,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo ; encoding: [0xf0,0x90,0x00,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo ; encoding: [0xc1,0x90,0x00,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, v4, v2 :: v_dual_fmac_f32 v9, v1, v3
// GFX1250: v_dual_fmac_f32 v7, v4, v2 :: v_dual_fmac_f32 v9, v1, v3 ; encoding: [0x04,0x01,0x00,0xcf,0x01,0x01,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, v1, v2 :: v_dual_fmac_f32 v9, v255, v3
// GFX1250: v_dual_fmac_f32 v7, v1, v2 :: v_dual_fmac_f32 v9, v255, v3 ; encoding: [0x01,0x01,0x00,0xcf,0xff,0x01,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, v255, v2 :: v_dual_fmac_f32 v9, v2, v3
// GFX1250: v_dual_fmac_f32 v7, v255, v2 :: v_dual_fmac_f32 v9, v2, v3 ; encoding: [0xff,0x01,0x00,0xcf,0x02,0x01,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, v2, v2 :: v_dual_fmac_f32 v9, v3, v3
// GFX1250: v_dual_fmac_f32 v7, v2, v2 :: v_dual_fmac_f32 v9, v3, v3 ; encoding: [0x02,0x01,0x00,0xcf,0x03,0x01,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, v3, v2 :: v_dual_fmac_f32 v9, v4, v3
// GFX1250: v_dual_fmac_f32 v7, v3, v2 :: v_dual_fmac_f32 v9, v4, v3 ; encoding: [0x03,0x01,0x00,0xcf,0x04,0x01,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, s105, v2 :: v_dual_fmac_f32 v9, s1, v3
// GFX1250: v_dual_fmac_f32 v7, s105, v2 :: v_dual_fmac_f32 v9, s1, v3 ; encoding: [0x69,0x00,0x00,0xcf,0x01,0x00,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, s1, v2 :: v_dual_fmac_f32 v9, s105, v3
// GFX1250: v_dual_fmac_f32 v7, s1, v2 :: v_dual_fmac_f32 v9, s105, v3 ; encoding: [0x01,0x00,0x00,0xcf,0x69,0x00,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, ttmp15, v2 :: v_dual_fmac_f32 v9, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v7, ttmp15, v2 :: v_dual_fmac_f32 v9, vcc_lo, v3 ; encoding: [0x7b,0x00,0x00,0xcf,0x6a,0x00,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, exec_hi, v2 :: v_dual_fmac_f32 v9, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v7, exec_hi, v2 :: v_dual_fmac_f32 v9, vcc_hi, v3 ; encoding: [0x7f,0x00,0x00,0xcf,0x6b,0x00,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, exec_lo, v2 :: v_dual_fmac_f32 v9, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v7, exec_lo, v2 :: v_dual_fmac_f32 v9, ttmp15, v3 ; encoding: [0x7e,0x00,0x00,0xcf,0x7b,0x00,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, m0, v2 :: v_dual_fmac_f32 v9, m0, v3
// GFX1250: v_dual_fmac_f32 v7, m0, v2 :: v_dual_fmac_f32 v9, m0, v3 ; encoding: [0x7d,0x00,0x00,0xcf,0x7d,0x00,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, vcc_hi, v2 :: v_dual_fmac_f32 v9, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v7, vcc_hi, v2 :: v_dual_fmac_f32 v9, exec_lo, v3 ; encoding: [0x6b,0x00,0x00,0xcf,0x7e,0x00,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, vcc_lo, v2 :: v_dual_fmac_f32 v9, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v7, vcc_lo, v2 :: v_dual_fmac_f32 v9, exec_hi, v3 ; encoding: [0x6a,0x00,0x00,0xcf,0x7f,0x00,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, src_scc, v2 :: v_dual_fmac_f32 v9, -1, v3
// GFX1250: v_dual_fmac_f32 v7, src_scc, v2 :: v_dual_fmac_f32 v9, -1, v3 ; encoding: [0xfd,0x00,0x00,0xcf,0xc1,0x00,0x02,0x00,0x07,0x03,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, 0.5, v3 :: v_dual_fmac_f32 v9, 0.5, v2
// GFX1250: v_dual_fmac_f32 v7, 0.5, v3 :: v_dual_fmac_f32 v9, 0.5, v2 ; encoding: [0xf0,0x00,0x00,0xcf,0xf0,0x00,0x03,0x00,0x07,0x02,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v7, -1, v4 :: v_dual_fmac_f32 v9, src_scc, v5
// GFX1250: v_dual_fmac_f32 v7, -1, v4 :: v_dual_fmac_f32 v9, src_scc, v5 ; encoding: [0xc1,0x00,0x00,0xcf,0xfd,0x00,0x04,0x00,0x07,0x05,0x00,0x09]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3 ; encoding: [0x04,0x11,0x01,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3 ; encoding: [0x01,0x11,0x01,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3 ; encoding: [0xff,0x11,0x01,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3 ; encoding: [0x02,0x11,0x01,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3 ; encoding: [0x03,0x11,0x01,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3 ; encoding: [0x69,0x10,0x01,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3 ; encoding: [0x01,0x10,0x01,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3 ; encoding: [0x7b,0x10,0x01,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3 ; encoding: [0x7f,0x10,0x01,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3 ; encoding: [0x7e,0x10,0x01,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3 ; encoding: [0x7d,0x10,0x01,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3 ; encoding: [0x6b,0x10,0x01,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3 ; encoding: [0x6a,0x10,0x01,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3 ; encoding: [0xfd,0x10,0x01,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2 ; encoding: [0xf0,0x10,0x01,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5 ; encoding: [0xc1,0x10,0x01,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3 ; encoding: [0x04,0xa1,0x00,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3 ; encoding: [0x01,0xa1,0x00,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3 ; encoding: [0xff,0xa1,0x00,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3 ; encoding: [0x02,0xa1,0x00,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3 ; encoding: [0x03,0xa1,0x00,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3 ; encoding: [0x69,0xa0,0x00,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3 ; encoding: [0x01,0xa0,0x00,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xa0,0x00,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xa0,0x00,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xa0,0x00,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3 ; encoding: [0x7d,0xa0,0x00,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xa0,0x00,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xa0,0x00,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3 ; encoding: [0xfd,0xa0,0x00,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xa0,0x00,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xa0,0x00,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3 ; encoding: [0x04,0xb1,0x00,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3 ; encoding: [0x01,0xb1,0x00,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3 ; encoding: [0xff,0xb1,0x00,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3 ; encoding: [0x02,0xb1,0x00,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3 ; encoding: [0x03,0xb1,0x00,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3 ; encoding: [0x69,0xb0,0x00,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3 ; encoding: [0x01,0xb0,0x00,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xb0,0x00,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xb0,0x00,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xb0,0x00,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3 ; encoding: [0x7d,0xb0,0x00,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xb0,0x00,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xb0,0x00,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3 ; encoding: [0xfd,0xb0,0x00,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xb0,0x00,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xb0,0x00,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1
// GFX1250: v_dual_fmac_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1 ; encoding: [0x04,0x81,0x00,0xcf,0x01,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255
// GFX1250: v_dual_fmac_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255 ; encoding: [0x01,0x81,0x00,0xcf,0xff,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2
// GFX1250: v_dual_fmac_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2 ; encoding: [0xff,0x81,0x00,0xcf,0x02,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3 ; encoding: [0x02,0x81,0x00,0xcf,0x03,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4
// GFX1250: v_dual_fmac_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4 ; encoding: [0x03,0x81,0x00,0xcf,0x04,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1
// GFX1250: v_dual_fmac_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1 ; encoding: [0x69,0x80,0x00,0xcf,0x01,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105
// GFX1250: v_dual_fmac_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105 ; encoding: [0x01,0x80,0x00,0xcf,0x69,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo ; encoding: [0x7b,0x80,0x00,0xcf,0x6a,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi ; encoding: [0x7f,0x80,0x00,0xcf,0x6b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15 ; encoding: [0x7e,0x80,0x00,0xcf,0x7b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0
// GFX1250: v_dual_fmac_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0 ; encoding: [0x7d,0x80,0x00,0xcf,0x7d,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo ; encoding: [0x6b,0x80,0x00,0xcf,0x7e,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi ; encoding: [0x6a,0x80,0x00,0xcf,0x7f,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1
// GFX1250: v_dual_fmac_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1 ; encoding: [0xfd,0x80,0x00,0xcf,0xc1,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5 ; encoding: [0xf0,0x80,0x00,0xcf,0xf0,0x00,0x03,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc ; encoding: [0xc1,0x80,0x00,0xcf,0xfd,0x00,0x04,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3 ; encoding: [0x04,0x71,0x00,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3 ; encoding: [0x01,0x71,0x00,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3 ; encoding: [0xff,0x71,0x00,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3 ; encoding: [0x02,0x71,0x00,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3 ; encoding: [0x03,0x71,0x00,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3 ; encoding: [0x69,0x70,0x00,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3 ; encoding: [0x01,0x70,0x00,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x00,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x00,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x00,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3 ; encoding: [0x7d,0x70,0x00,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x00,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x00,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3 ; encoding: [0xfd,0x70,0x00,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x00,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x00,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3 ; encoding: [0x04,0x31,0x00,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3 ; encoding: [0x01,0x31,0x00,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3 ; encoding: [0xff,0x31,0x00,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3 ; encoding: [0x02,0x31,0x00,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3 ; encoding: [0x03,0x31,0x00,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3 ; encoding: [0x69,0x30,0x00,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3 ; encoding: [0x01,0x30,0x00,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x30,0x00,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x30,0x00,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x30,0x00,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3 ; encoding: [0x7d,0x30,0x00,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x30,0x00,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x30,0x00,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3 ; encoding: [0xfd,0x30,0x00,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2 ; encoding: [0xf0,0x30,0x00,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5 ; encoding: [0xc1,0x30,0x00,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3 ; encoding: [0x04,0x51,0x00,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3 ; encoding: [0x01,0x51,0x00,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3 ; encoding: [0xff,0x51,0x00,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3 ; encoding: [0x02,0x51,0x00,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3 ; encoding: [0x03,0x51,0x00,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3 ; encoding: [0x69,0x50,0x00,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3 ; encoding: [0x01,0x50,0x00,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x50,0x00,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x50,0x00,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x50,0x00,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3 ; encoding: [0x7d,0x50,0x00,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x50,0x00,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x50,0x00,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3 ; encoding: [0xfd,0x50,0x00,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2 ; encoding: [0xf0,0x50,0x00,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5 ; encoding: [0xc1,0x50,0x00,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3 ; encoding: [0x04,0x61,0x00,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3 ; encoding: [0x01,0x61,0x00,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3 ; encoding: [0xff,0x61,0x00,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3 ; encoding: [0x02,0x61,0x00,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3 ; encoding: [0x03,0x61,0x00,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3 ; encoding: [0x69,0x60,0x00,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3 ; encoding: [0x01,0x60,0x00,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x60,0x00,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x60,0x00,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x60,0x00,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3 ; encoding: [0x7d,0x60,0x00,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x60,0x00,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x60,0x00,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3 ; encoding: [0xfd,0x60,0x00,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2 ; encoding: [0xf0,0x60,0x00,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5 ; encoding: [0xc1,0x60,0x00,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4 ; encoding: [0x04,0x31,0x01,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x04,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:20
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:0x14 ; encoding: [0x04,0x21,0x01,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x14,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3 ; encoding: [0x04,0x41,0x28,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3 ; encoding: [0x01,0x41,0x28,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3 ; encoding: [0xff,0x41,0x28,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3 ; encoding: [0x02,0x41,0x28,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3 ; encoding: [0x03,0x41,0x28,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3 ; encoding: [0x69,0x40,0x28,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3 ; encoding: [0x01,0x40,0x28,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x28,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x28,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x28,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3 ; encoding: [0x7d,0x40,0x28,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x28,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x28,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3 ; encoding: [0xfd,0x40,0x28,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x28,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x28,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3 ; encoding: [0x04,0x01,0x29,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3 ; encoding: [0x01,0x01,0x29,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3 ; encoding: [0xff,0x01,0x29,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3 ; encoding: [0x02,0x01,0x29,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3 ; encoding: [0x03,0x01,0x29,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3 ; encoding: [0x69,0x00,0x29,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3 ; encoding: [0x01,0x00,0x29,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x29,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x29,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x29,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x00,0x29,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x29,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x29,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x00,0x29,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x29,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x29,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo ; encoding: [0x04,0x91,0x28,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo ; encoding: [0x01,0x91,0x28,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo ; encoding: [0xff,0x91,0x28,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo ; encoding: [0x02,0x91,0x28,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo ; encoding: [0x03,0x91,0x28,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo ; encoding: [0x69,0x90,0x28,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo ; encoding: [0x01,0x90,0x28,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo ; encoding: [0x7b,0x90,0x28,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo ; encoding: [0x7f,0x90,0x28,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo ; encoding: [0x7e,0x90,0x28,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo ; encoding: [0x7d,0x90,0x28,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo ; encoding: [0x6b,0x90,0x28,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo ; encoding: [0x6a,0x90,0x28,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo ; encoding: [0xfd,0x90,0x28,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo ; encoding: [0xf0,0x90,0x28,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo ; encoding: [0xc1,0x90,0x28,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3 ; encoding: [0x04,0x01,0x28,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3 ; encoding: [0x01,0x01,0x28,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3 ; encoding: [0xff,0x01,0x28,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3 ; encoding: [0x02,0x01,0x28,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3 ; encoding: [0x03,0x01,0x28,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3 ; encoding: [0x69,0x00,0x28,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3 ; encoding: [0x01,0x00,0x28,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x28,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x28,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x28,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3 ; encoding: [0x7d,0x00,0x28,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x28,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x28,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3 ; encoding: [0xfd,0x00,0x28,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x28,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x28,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3 ; encoding: [0x04,0x11,0x29,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3 ; encoding: [0x01,0x11,0x29,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3 ; encoding: [0xff,0x11,0x29,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3 ; encoding: [0x02,0x11,0x29,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3 ; encoding: [0x03,0x11,0x29,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3 ; encoding: [0x69,0x10,0x29,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3 ; encoding: [0x01,0x10,0x29,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3 ; encoding: [0x7b,0x10,0x29,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3 ; encoding: [0x7f,0x10,0x29,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3 ; encoding: [0x7e,0x10,0x29,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3 ; encoding: [0x7d,0x10,0x29,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3 ; encoding: [0x6b,0x10,0x29,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3 ; encoding: [0x6a,0x10,0x29,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3 ; encoding: [0xfd,0x10,0x29,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2 ; encoding: [0xf0,0x10,0x29,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5 ; encoding: [0xc1,0x10,0x29,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3 ; encoding: [0x04,0xa1,0x28,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3 ; encoding: [0x01,0xa1,0x28,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3 ; encoding: [0xff,0xa1,0x28,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3 ; encoding: [0x02,0xa1,0x28,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3 ; encoding: [0x03,0xa1,0x28,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3 ; encoding: [0x69,0xa0,0x28,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3 ; encoding: [0x01,0xa0,0x28,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xa0,0x28,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xa0,0x28,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xa0,0x28,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3 ; encoding: [0x7d,0xa0,0x28,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xa0,0x28,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xa0,0x28,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3 ; encoding: [0xfd,0xa0,0x28,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xa0,0x28,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xa0,0x28,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3 ; encoding: [0x04,0xb1,0x28,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3 ; encoding: [0x01,0xb1,0x28,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3 ; encoding: [0xff,0xb1,0x28,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3 ; encoding: [0x02,0xb1,0x28,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3 ; encoding: [0x03,0xb1,0x28,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3 ; encoding: [0x69,0xb0,0x28,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3 ; encoding: [0x01,0xb0,0x28,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xb0,0x28,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xb0,0x28,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xb0,0x28,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3 ; encoding: [0x7d,0xb0,0x28,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xb0,0x28,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xb0,0x28,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3 ; encoding: [0xfd,0xb0,0x28,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xb0,0x28,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xb0,0x28,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1
// GFX1250: v_dual_max_num_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1 ; encoding: [0x04,0x81,0x28,0xcf,0x01,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255
// GFX1250: v_dual_max_num_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255 ; encoding: [0x01,0x81,0x28,0xcf,0xff,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2
// GFX1250: v_dual_max_num_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2 ; encoding: [0xff,0x81,0x28,0xcf,0x02,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3 ; encoding: [0x02,0x81,0x28,0xcf,0x03,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4
// GFX1250: v_dual_max_num_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4 ; encoding: [0x03,0x81,0x28,0xcf,0x04,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1
// GFX1250: v_dual_max_num_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1 ; encoding: [0x69,0x80,0x28,0xcf,0x01,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105
// GFX1250: v_dual_max_num_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105 ; encoding: [0x01,0x80,0x28,0xcf,0x69,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo ; encoding: [0x7b,0x80,0x28,0xcf,0x6a,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi ; encoding: [0x7f,0x80,0x28,0xcf,0x6b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15 ; encoding: [0x7e,0x80,0x28,0xcf,0x7b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0
// GFX1250: v_dual_max_num_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0 ; encoding: [0x7d,0x80,0x28,0xcf,0x7d,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo ; encoding: [0x6b,0x80,0x28,0xcf,0x7e,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi ; encoding: [0x6a,0x80,0x28,0xcf,0x7f,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1
// GFX1250: v_dual_max_num_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1 ; encoding: [0xfd,0x80,0x28,0xcf,0xc1,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5 ; encoding: [0xf0,0x80,0x28,0xcf,0xf0,0x00,0x03,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc ; encoding: [0xc1,0x80,0x28,0xcf,0xfd,0x00,0x04,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3 ; encoding: [0x04,0x71,0x28,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3 ; encoding: [0x01,0x71,0x28,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3 ; encoding: [0xff,0x71,0x28,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3 ; encoding: [0x02,0x71,0x28,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3 ; encoding: [0x03,0x71,0x28,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3 ; encoding: [0x69,0x70,0x28,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3 ; encoding: [0x01,0x70,0x28,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x28,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x28,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x28,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3 ; encoding: [0x7d,0x70,0x28,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x28,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x28,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3 ; encoding: [0xfd,0x70,0x28,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x28,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x28,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3 ; encoding: [0x04,0x31,0x28,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3 ; encoding: [0x01,0x31,0x28,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3 ; encoding: [0xff,0x31,0x28,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3 ; encoding: [0x02,0x31,0x28,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3 ; encoding: [0x03,0x31,0x28,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3 ; encoding: [0x69,0x30,0x28,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3 ; encoding: [0x01,0x30,0x28,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x30,0x28,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x30,0x28,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x30,0x28,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3 ; encoding: [0x7d,0x30,0x28,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x30,0x28,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x30,0x28,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3 ; encoding: [0xfd,0x30,0x28,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2 ; encoding: [0xf0,0x30,0x28,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5 ; encoding: [0xc1,0x30,0x28,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3 ; encoding: [0x04,0x51,0x28,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3 ; encoding: [0x01,0x51,0x28,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3 ; encoding: [0xff,0x51,0x28,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3 ; encoding: [0x02,0x51,0x28,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3 ; encoding: [0x03,0x51,0x28,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3 ; encoding: [0x69,0x50,0x28,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3 ; encoding: [0x01,0x50,0x28,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x50,0x28,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x50,0x28,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x50,0x28,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3 ; encoding: [0x7d,0x50,0x28,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x50,0x28,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x50,0x28,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3 ; encoding: [0xfd,0x50,0x28,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2 ; encoding: [0xf0,0x50,0x28,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5 ; encoding: [0xc1,0x50,0x28,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3 ; encoding: [0x04,0x61,0x28,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3 ; encoding: [0x01,0x61,0x28,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3 ; encoding: [0xff,0x61,0x28,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3 ; encoding: [0x02,0x61,0x28,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3 ; encoding: [0x03,0x61,0x28,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3 ; encoding: [0x69,0x60,0x28,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3 ; encoding: [0x01,0x60,0x28,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x60,0x28,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x60,0x28,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x60,0x28,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3 ; encoding: [0x7d,0x60,0x28,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x60,0x28,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x60,0x28,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3 ; encoding: [0xfd,0x60,0x28,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2 ; encoding: [0xf0,0x60,0x28,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5 ; encoding: [0xc1,0x60,0x28,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4 ; encoding: [0x04,0x31,0x29,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x04,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:0x6e
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:0x6e ; encoding: [0x04,0x21,0x29,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x6e,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3 ; encoding: [0x04,0x41,0x2c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3 ; encoding: [0x01,0x41,0x2c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3 ; encoding: [0xff,0x41,0x2c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3 ; encoding: [0x02,0x41,0x2c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3 ; encoding: [0x03,0x41,0x2c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3 ; encoding: [0x69,0x40,0x2c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3 ; encoding: [0x01,0x40,0x2c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x2c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x2c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x2c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3 ; encoding: [0x7d,0x40,0x2c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x2c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x2c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3 ; encoding: [0xfd,0x40,0x2c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x2c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x2c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3 ; encoding: [0x04,0x01,0x2d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3 ; encoding: [0x01,0x01,0x2d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3 ; encoding: [0xff,0x01,0x2d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3 ; encoding: [0x02,0x01,0x2d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3 ; encoding: [0x03,0x01,0x2d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3 ; encoding: [0x69,0x00,0x2d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3 ; encoding: [0x01,0x00,0x2d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x2d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x2d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x2d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x00,0x2d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x2d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x2d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x00,0x2d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x2d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x2d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo ; encoding: [0x04,0x91,0x2c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo ; encoding: [0x01,0x91,0x2c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo ; encoding: [0xff,0x91,0x2c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo ; encoding: [0x02,0x91,0x2c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo ; encoding: [0x03,0x91,0x2c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo ; encoding: [0x69,0x90,0x2c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo ; encoding: [0x01,0x90,0x2c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo ; encoding: [0x7b,0x90,0x2c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo ; encoding: [0x7f,0x90,0x2c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo ; encoding: [0x7e,0x90,0x2c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo ; encoding: [0x7d,0x90,0x2c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo ; encoding: [0x6b,0x90,0x2c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo ; encoding: [0x6a,0x90,0x2c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo ; encoding: [0xfd,0x90,0x2c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo ; encoding: [0xf0,0x90,0x2c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo ; encoding: [0xc1,0x90,0x2c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3 ; encoding: [0x04,0x01,0x2c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3 ; encoding: [0x01,0x01,0x2c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3 ; encoding: [0xff,0x01,0x2c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3 ; encoding: [0x02,0x01,0x2c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3 ; encoding: [0x03,0x01,0x2c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3 ; encoding: [0x69,0x00,0x2c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3 ; encoding: [0x01,0x00,0x2c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x2c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x2c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x2c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3 ; encoding: [0x7d,0x00,0x2c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x2c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x2c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3 ; encoding: [0xfd,0x00,0x2c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x2c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x2c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3 ; encoding: [0x04,0x11,0x2d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3 ; encoding: [0x01,0x11,0x2d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3 ; encoding: [0xff,0x11,0x2d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3 ; encoding: [0x02,0x11,0x2d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3 ; encoding: [0x03,0x11,0x2d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3 ; encoding: [0x69,0x10,0x2d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3 ; encoding: [0x01,0x10,0x2d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3 ; encoding: [0x7b,0x10,0x2d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3 ; encoding: [0x7f,0x10,0x2d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3 ; encoding: [0x7e,0x10,0x2d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3 ; encoding: [0x7d,0x10,0x2d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3 ; encoding: [0x6b,0x10,0x2d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3 ; encoding: [0x6a,0x10,0x2d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3 ; encoding: [0xfd,0x10,0x2d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2 ; encoding: [0xf0,0x10,0x2d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5 ; encoding: [0xc1,0x10,0x2d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3 ; encoding: [0x04,0xa1,0x2c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3 ; encoding: [0x01,0xa1,0x2c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3 ; encoding: [0xff,0xa1,0x2c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3 ; encoding: [0x02,0xa1,0x2c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3 ; encoding: [0x03,0xa1,0x2c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3 ; encoding: [0x69,0xa0,0x2c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3 ; encoding: [0x01,0xa0,0x2c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xa0,0x2c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xa0,0x2c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xa0,0x2c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3 ; encoding: [0x7d,0xa0,0x2c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xa0,0x2c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xa0,0x2c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3 ; encoding: [0xfd,0xa0,0x2c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xa0,0x2c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xa0,0x2c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3 ; encoding: [0x04,0xb1,0x2c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3 ; encoding: [0x01,0xb1,0x2c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3 ; encoding: [0xff,0xb1,0x2c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3 ; encoding: [0x02,0xb1,0x2c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3 ; encoding: [0x03,0xb1,0x2c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3 ; encoding: [0x69,0xb0,0x2c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3 ; encoding: [0x01,0xb0,0x2c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xb0,0x2c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xb0,0x2c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xb0,0x2c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3 ; encoding: [0x7d,0xb0,0x2c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xb0,0x2c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xb0,0x2c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3 ; encoding: [0xfd,0xb0,0x2c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xb0,0x2c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xb0,0x2c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1
// GFX1250: v_dual_min_num_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1 ; encoding: [0x04,0x81,0x2c,0xcf,0x01,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255
// GFX1250: v_dual_min_num_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255 ; encoding: [0x01,0x81,0x2c,0xcf,0xff,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2
// GFX1250: v_dual_min_num_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2 ; encoding: [0xff,0x81,0x2c,0xcf,0x02,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3 ; encoding: [0x02,0x81,0x2c,0xcf,0x03,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4
// GFX1250: v_dual_min_num_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4 ; encoding: [0x03,0x81,0x2c,0xcf,0x04,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1
// GFX1250: v_dual_min_num_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1 ; encoding: [0x69,0x80,0x2c,0xcf,0x01,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105
// GFX1250: v_dual_min_num_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105 ; encoding: [0x01,0x80,0x2c,0xcf,0x69,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo ; encoding: [0x7b,0x80,0x2c,0xcf,0x6a,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi ; encoding: [0x7f,0x80,0x2c,0xcf,0x6b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15 ; encoding: [0x7e,0x80,0x2c,0xcf,0x7b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0
// GFX1250: v_dual_min_num_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0 ; encoding: [0x7d,0x80,0x2c,0xcf,0x7d,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo ; encoding: [0x6b,0x80,0x2c,0xcf,0x7e,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi ; encoding: [0x6a,0x80,0x2c,0xcf,0x7f,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1
// GFX1250: v_dual_min_num_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1 ; encoding: [0xfd,0x80,0x2c,0xcf,0xc1,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5 ; encoding: [0xf0,0x80,0x2c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc ; encoding: [0xc1,0x80,0x2c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3 ; encoding: [0x04,0x71,0x2c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3 ; encoding: [0x01,0x71,0x2c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3 ; encoding: [0xff,0x71,0x2c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3 ; encoding: [0x02,0x71,0x2c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3 ; encoding: [0x03,0x71,0x2c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3 ; encoding: [0x69,0x70,0x2c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3 ; encoding: [0x01,0x70,0x2c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x2c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x2c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x2c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3 ; encoding: [0x7d,0x70,0x2c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x2c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x2c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3 ; encoding: [0xfd,0x70,0x2c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x2c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x2c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3 ; encoding: [0x04,0x31,0x2c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3 ; encoding: [0x01,0x31,0x2c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3 ; encoding: [0xff,0x31,0x2c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3 ; encoding: [0x02,0x31,0x2c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3 ; encoding: [0x03,0x31,0x2c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3 ; encoding: [0x69,0x30,0x2c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3 ; encoding: [0x01,0x30,0x2c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x30,0x2c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x30,0x2c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x30,0x2c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3 ; encoding: [0x7d,0x30,0x2c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x30,0x2c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x30,0x2c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3 ; encoding: [0xfd,0x30,0x2c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2 ; encoding: [0xf0,0x30,0x2c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5 ; encoding: [0xc1,0x30,0x2c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3 ; encoding: [0x04,0x51,0x2c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3 ; encoding: [0x01,0x51,0x2c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3 ; encoding: [0xff,0x51,0x2c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3 ; encoding: [0x02,0x51,0x2c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3 ; encoding: [0x03,0x51,0x2c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3 ; encoding: [0x69,0x50,0x2c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3 ; encoding: [0x01,0x50,0x2c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x50,0x2c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x50,0x2c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x50,0x2c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3 ; encoding: [0x7d,0x50,0x2c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x50,0x2c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x50,0x2c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3 ; encoding: [0xfd,0x50,0x2c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2 ; encoding: [0xf0,0x50,0x2c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5 ; encoding: [0xc1,0x50,0x2c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3 ; encoding: [0x04,0x61,0x2c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3 ; encoding: [0x01,0x61,0x2c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3 ; encoding: [0xff,0x61,0x2c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3 ; encoding: [0x02,0x61,0x2c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3 ; encoding: [0x03,0x61,0x2c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3 ; encoding: [0x69,0x60,0x2c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3 ; encoding: [0x01,0x60,0x2c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x60,0x2c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x60,0x2c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x60,0x2c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3 ; encoding: [0x7d,0x60,0x2c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x60,0x2c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x60,0x2c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3 ; encoding: [0xfd,0x60,0x2c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2 ; encoding: [0xf0,0x60,0x2c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5 ; encoding: [0xc1,0x60,0x2c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4 ; encoding: [0x04,0x31,0x2d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x04,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:255
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:0xff ; encoding: [0x04,0x21,0x2d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0xff,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_add_f32 v7, v1, v255
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_add_f32 v7, v1, v255 ; encoding: [0x04,0x41,0x20,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_add_f32 v7, v255, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_add_f32 v7, v255, v255 ; encoding: [0x01,0x41,0x20,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_add_f32 v7, v2, v255
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_add_f32 v7, v2, v255 ; encoding: [0xff,0x41,0x20,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_add_f32 v7, v3, v255
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_add_f32 v7, v3, v255 ; encoding: [0x02,0x41,0x20,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_add_f32 v7, v4, v255
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_add_f32 v7, v4, v255 ; encoding: [0x03,0x41,0x20,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_add_f32 v7, s1, v255
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_add_f32 v7, s1, v255 ; encoding: [0x69,0x40,0x20,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_add_f32 v7, s105, v255
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_add_f32 v7, s105, v255 ; encoding: [0x01,0x40,0x20,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_add_f32 v7, vcc_lo, v255
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_add_f32 v7, vcc_lo, v255 ; encoding: [0x7b,0x40,0x20,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_add_f32 v7, vcc_hi, v255
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_add_f32 v7, vcc_hi, v255 ; encoding: [0x7f,0x40,0x20,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_add_f32 v7, ttmp15, v255
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_add_f32 v7, ttmp15, v255 ; encoding: [0x7e,0x40,0x20,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_add_f32 v7, m0, v255
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_add_f32 v7, m0, v255 ; encoding: [0x7d,0x40,0x20,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_add_f32 v7, exec_lo, v255
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_add_f32 v7, exec_lo, v255 ; encoding: [0x6b,0x40,0x20,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_add_f32 v7, exec_hi, v255
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_add_f32 v7, exec_hi, v255 ; encoding: [0x6a,0x40,0x20,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_add_f32 v7, -1, v255
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_add_f32 v7, -1, v255 ; encoding: [0xfd,0x40,0x20,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_add_f32 v7, 0.5, v3
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_add_f32 v7, 0.5, v3 ; encoding: [0xf0,0x40,0x20,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_add_f32 v7, src_scc, v4
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_add_f32 v7, src_scc, v4 ; encoding: [0xc1,0x40,0x20,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_add_nc_u32 v7, v1, v255
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_add_nc_u32 v7, v1, v255 ; encoding: [0x04,0x01,0x21,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_add_nc_u32 v7, v255, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_add_nc_u32 v7, v255, v255 ; encoding: [0x01,0x01,0x21,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_add_nc_u32 v7, v2, v255
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_add_nc_u32 v7, v2, v255 ; encoding: [0xff,0x01,0x21,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_add_nc_u32 v7, v3, v255
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_add_nc_u32 v7, v3, v255 ; encoding: [0x02,0x01,0x21,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_add_nc_u32 v7, v4, v255
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_add_nc_u32 v7, v4, v255 ; encoding: [0x03,0x01,0x21,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_add_nc_u32 v7, s1, v255
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_add_nc_u32 v7, s1, v255 ; encoding: [0x69,0x00,0x21,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_add_nc_u32 v7, s105, v255
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_add_nc_u32 v7, s105, v255 ; encoding: [0x01,0x00,0x21,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_add_nc_u32 v7, vcc_lo, v255
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_add_nc_u32 v7, vcc_lo, v255 ; encoding: [0x7b,0x00,0x21,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_add_nc_u32 v7, vcc_hi, v255
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_add_nc_u32 v7, vcc_hi, v255 ; encoding: [0x7f,0x00,0x21,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_add_nc_u32 v7, ttmp15, v255
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_add_nc_u32 v7, ttmp15, v255 ; encoding: [0x7e,0x00,0x21,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_add_nc_u32 v7, m0, v255
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_add_nc_u32 v7, m0, v255 ; encoding: [0x7d,0x00,0x21,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_add_nc_u32 v7, exec_lo, v255
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_add_nc_u32 v7, exec_lo, v255 ; encoding: [0x6b,0x00,0x21,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_add_nc_u32 v7, exec_hi, v255
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_add_nc_u32 v7, exec_hi, v255 ; encoding: [0x6a,0x00,0x21,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_add_nc_u32 v7, -1, v255
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_add_nc_u32 v7, -1, v255 ; encoding: [0xfd,0x00,0x21,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_add_nc_u32 v7, 0.5, v3
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_add_nc_u32 v7, 0.5, v3 ; encoding: [0xf0,0x00,0x21,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_add_nc_u32 v7, src_scc, v4
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_add_nc_u32 v7, src_scc, v4 ; encoding: [0xc1,0x00,0x21,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_cndmask_b32 v7, v1, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_cndmask_b32 v7, v1, v255, vcc_lo ; encoding: [0x04,0x91,0x20,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_cndmask_b32 v7, v255, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_cndmask_b32 v7, v255, v255, vcc_lo ; encoding: [0x01,0x91,0x20,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_cndmask_b32 v7, v2, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_cndmask_b32 v7, v2, v255, vcc_lo ; encoding: [0xff,0x91,0x20,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_cndmask_b32 v7, v3, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_cndmask_b32 v7, v3, v255, vcc_lo ; encoding: [0x02,0x91,0x20,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_cndmask_b32 v7, v4, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_cndmask_b32 v7, v4, v255, vcc_lo ; encoding: [0x03,0x91,0x20,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_cndmask_b32 v7, s105, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_cndmask_b32 v7, s105, v255, vcc_lo ; encoding: [0x69,0x90,0x20,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_cndmask_b32 v7, s1, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_cndmask_b32 v7, s1, v255, vcc_lo ; encoding: [0x01,0x90,0x20,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_cndmask_b32 v7, ttmp15, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_cndmask_b32 v7, ttmp15, v255, vcc_lo ; encoding: [0x7b,0x90,0x20,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_cndmask_b32 v7, exec_hi, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_cndmask_b32 v7, exec_hi, v255, vcc_lo ; encoding: [0x7f,0x90,0x20,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_cndmask_b32 v7, exec_lo, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_cndmask_b32 v7, exec_lo, v255, vcc_lo ; encoding: [0x7e,0x90,0x20,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_cndmask_b32 v7, m0, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_cndmask_b32 v7, m0, v255, vcc_lo ; encoding: [0x7d,0x90,0x20,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_cndmask_b32 v7, vcc_hi, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_cndmask_b32 v7, vcc_hi, v255, vcc_lo ; encoding: [0x6b,0x90,0x20,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_cndmask_b32 v7, vcc_lo, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_cndmask_b32 v7, vcc_lo, v255, vcc_lo ; encoding: [0x6a,0x90,0x20,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_cndmask_b32 v7, -1, v255, vcc_lo
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_cndmask_b32 v7, -1, v255, vcc_lo ; encoding: [0xfd,0x90,0x20,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_cndmask_b32 v7, 0.5, v3, vcc_lo
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_cndmask_b32 v7, 0.5, v3, vcc_lo ; encoding: [0xf0,0x90,0x20,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_cndmask_b32 v7, src_scc, v4, vcc_lo
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_cndmask_b32 v7, src_scc, v4, vcc_lo ; encoding: [0xc1,0x90,0x20,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_fmac_f32 v7, v1, v255
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_fmac_f32 v7, v1, v255 ; encoding: [0x04,0x01,0x20,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_fmac_f32 v7, v255, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_fmac_f32 v7, v255, v255 ; encoding: [0x01,0x01,0x20,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_fmac_f32 v7, v2, v255
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_fmac_f32 v7, v2, v255 ; encoding: [0xff,0x01,0x20,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_fmac_f32 v7, v3, v255
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_fmac_f32 v7, v3, v255 ; encoding: [0x02,0x01,0x20,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_fmac_f32 v7, v4, v255
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_fmac_f32 v7, v4, v255 ; encoding: [0x03,0x01,0x20,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_fmac_f32 v7, s1, v255
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_fmac_f32 v7, s1, v255 ; encoding: [0x69,0x00,0x20,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_fmac_f32 v7, s105, v255
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_fmac_f32 v7, s105, v255 ; encoding: [0x01,0x00,0x20,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_fmac_f32 v7, vcc_lo, v255
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_fmac_f32 v7, vcc_lo, v255 ; encoding: [0x7b,0x00,0x20,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_fmac_f32 v7, vcc_hi, v255
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_fmac_f32 v7, vcc_hi, v255 ; encoding: [0x7f,0x00,0x20,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_fmac_f32 v7, ttmp15, v255
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_fmac_f32 v7, ttmp15, v255 ; encoding: [0x7e,0x00,0x20,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_fmac_f32 v7, m0, v255
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_fmac_f32 v7, m0, v255 ; encoding: [0x7d,0x00,0x20,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_fmac_f32 v7, exec_lo, v255
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_fmac_f32 v7, exec_lo, v255 ; encoding: [0x6b,0x00,0x20,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_fmac_f32 v7, exec_hi, v255
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_fmac_f32 v7, exec_hi, v255 ; encoding: [0x6a,0x00,0x20,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_fmac_f32 v7, -1, v255
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_fmac_f32 v7, -1, v255 ; encoding: [0xfd,0x00,0x20,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_fmac_f32 v7, 0.5, v3
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_fmac_f32 v7, 0.5, v3 ; encoding: [0xf0,0x00,0x20,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_fmac_f32 v7, src_scc, v4
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_fmac_f32 v7, src_scc, v4 ; encoding: [0xc1,0x00,0x20,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_lshlrev_b32 v7, v1, v255
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_lshlrev_b32 v7, v1, v255 ; encoding: [0x04,0x11,0x21,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_lshlrev_b32 v7, v255, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_lshlrev_b32 v7, v255, v255 ; encoding: [0x01,0x11,0x21,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_lshlrev_b32 v7, v2, v255
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_lshlrev_b32 v7, v2, v255 ; encoding: [0xff,0x11,0x21,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_lshlrev_b32 v7, v3, v255
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_lshlrev_b32 v7, v3, v255 ; encoding: [0x02,0x11,0x21,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_lshlrev_b32 v7, v4, v255
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_lshlrev_b32 v7, v4, v255 ; encoding: [0x03,0x11,0x21,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_lshlrev_b32 v7, s1, v255
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_lshlrev_b32 v7, s1, v255 ; encoding: [0x69,0x10,0x21,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_lshlrev_b32 v7, s105, v255
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_lshlrev_b32 v7, s105, v255 ; encoding: [0x01,0x10,0x21,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_lshlrev_b32 v7, vcc_lo, v255
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_lshlrev_b32 v7, vcc_lo, v255 ; encoding: [0x7b,0x10,0x21,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_lshlrev_b32 v7, vcc_hi, v255
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_lshlrev_b32 v7, vcc_hi, v255 ; encoding: [0x7f,0x10,0x21,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_lshlrev_b32 v7, ttmp15, v255
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_lshlrev_b32 v7, ttmp15, v255 ; encoding: [0x7e,0x10,0x21,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_lshlrev_b32 v7, m0, v255
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_lshlrev_b32 v7, m0, v255 ; encoding: [0x7d,0x10,0x21,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_lshlrev_b32 v7, exec_lo, v255
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_lshlrev_b32 v7, exec_lo, v255 ; encoding: [0x6b,0x10,0x21,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_lshlrev_b32 v7, exec_hi, v255
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_lshlrev_b32 v7, exec_hi, v255 ; encoding: [0x6a,0x10,0x21,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_lshlrev_b32 v7, -1, v255
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_lshlrev_b32 v7, -1, v255 ; encoding: [0xfd,0x10,0x21,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_lshlrev_b32 v7, 0.5, v3
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_lshlrev_b32 v7, 0.5, v3 ; encoding: [0xf0,0x10,0x21,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_lshlrev_b32 v7, src_scc, v4
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_lshlrev_b32 v7, src_scc, v4 ; encoding: [0xc1,0x10,0x21,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_max_num_f32 v7, v1, v255
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_max_num_f32 v7, v1, v255 ; encoding: [0x04,0xa1,0x20,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_max_num_f32 v7, v255, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_max_num_f32 v7, v255, v255 ; encoding: [0x01,0xa1,0x20,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_max_num_f32 v7, v2, v255
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_max_num_f32 v7, v2, v255 ; encoding: [0xff,0xa1,0x20,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_max_num_f32 v7, v3, v255
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_max_num_f32 v7, v3, v255 ; encoding: [0x02,0xa1,0x20,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_max_num_f32 v7, v4, v255
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_max_num_f32 v7, v4, v255 ; encoding: [0x03,0xa1,0x20,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_max_num_f32 v7, s1, v255
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_max_num_f32 v7, s1, v255 ; encoding: [0x69,0xa0,0x20,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_max_num_f32 v7, s105, v255
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_max_num_f32 v7, s105, v255 ; encoding: [0x01,0xa0,0x20,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_max_num_f32 v7, vcc_lo, v255
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_max_num_f32 v7, vcc_lo, v255 ; encoding: [0x7b,0xa0,0x20,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_max_num_f32 v7, vcc_hi, v255
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_max_num_f32 v7, vcc_hi, v255 ; encoding: [0x7f,0xa0,0x20,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_max_num_f32 v7, ttmp15, v255
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_max_num_f32 v7, ttmp15, v255 ; encoding: [0x7e,0xa0,0x20,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_max_num_f32 v7, m0, v255
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_max_num_f32 v7, m0, v255 ; encoding: [0x7d,0xa0,0x20,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_max_num_f32 v7, exec_lo, v255
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_max_num_f32 v7, exec_lo, v255 ; encoding: [0x6b,0xa0,0x20,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_max_num_f32 v7, exec_hi, v255
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_max_num_f32 v7, exec_hi, v255 ; encoding: [0x6a,0xa0,0x20,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_max_num_f32 v7, -1, v255
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_max_num_f32 v7, -1, v255 ; encoding: [0xfd,0xa0,0x20,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_max_num_f32 v7, 0.5, v3
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_max_num_f32 v7, 0.5, v3 ; encoding: [0xf0,0xa0,0x20,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_max_num_f32 v7, src_scc, v4
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_max_num_f32 v7, src_scc, v4 ; encoding: [0xc1,0xa0,0x20,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_min_num_f32 v7, v1, v255
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_min_num_f32 v7, v1, v255 ; encoding: [0x04,0xb1,0x20,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_min_num_f32 v7, v255, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_min_num_f32 v7, v255, v255 ; encoding: [0x01,0xb1,0x20,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_min_num_f32 v7, v2, v255
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_min_num_f32 v7, v2, v255 ; encoding: [0xff,0xb1,0x20,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_min_num_f32 v7, v3, v255
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_min_num_f32 v7, v3, v255 ; encoding: [0x02,0xb1,0x20,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_min_num_f32 v7, v4, v255
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_min_num_f32 v7, v4, v255 ; encoding: [0x03,0xb1,0x20,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_min_num_f32 v7, s1, v255
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_min_num_f32 v7, s1, v255 ; encoding: [0x69,0xb0,0x20,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_min_num_f32 v7, s105, v255
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_min_num_f32 v7, s105, v255 ; encoding: [0x01,0xb0,0x20,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_min_num_f32 v7, vcc_lo, v255
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_min_num_f32 v7, vcc_lo, v255 ; encoding: [0x7b,0xb0,0x20,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_min_num_f32 v7, vcc_hi, v255
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_min_num_f32 v7, vcc_hi, v255 ; encoding: [0x7f,0xb0,0x20,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_min_num_f32 v7, ttmp15, v255
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_min_num_f32 v7, ttmp15, v255 ; encoding: [0x7e,0xb0,0x20,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_min_num_f32 v7, m0, v255
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_min_num_f32 v7, m0, v255 ; encoding: [0x7d,0xb0,0x20,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_min_num_f32 v7, exec_lo, v255
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_min_num_f32 v7, exec_lo, v255 ; encoding: [0x6b,0xb0,0x20,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_min_num_f32 v7, exec_hi, v255
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_min_num_f32 v7, exec_hi, v255 ; encoding: [0x6a,0xb0,0x20,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_min_num_f32 v7, -1, v255
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_min_num_f32 v7, -1, v255 ; encoding: [0xfd,0xb0,0x20,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_min_num_f32 v7, 0.5, v3
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_min_num_f32 v7, 0.5, v3 ; encoding: [0xf0,0xb0,0x20,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_min_num_f32 v7, src_scc, v4
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_min_num_f32 v7, src_scc, v4 ; encoding: [0xc1,0xb0,0x20,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_mov_b32 v7, v1
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_mov_b32 v7, v1 ; encoding: [0x04,0x81,0x20,0xcf,0x01,0x01,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_mov_b32 v7, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_mov_b32 v7, v255 ; encoding: [0x01,0x81,0x20,0xcf,0xff,0x01,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_mov_b32 v7, v2
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_mov_b32 v7, v2 ; encoding: [0xff,0x81,0x20,0xcf,0x02,0x01,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_mov_b32 v7, v3
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_mov_b32 v7, v3 ; encoding: [0x02,0x81,0x20,0xcf,0x03,0x01,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_mov_b32 v7, v4
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_mov_b32 v7, v4 ; encoding: [0x03,0x81,0x20,0xcf,0x04,0x01,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_mov_b32 v7, s1
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_mov_b32 v7, s1 ; encoding: [0x69,0x80,0x20,0xcf,0x01,0x00,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_mov_b32 v7, s105
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_mov_b32 v7, s105 ; encoding: [0x01,0x80,0x20,0xcf,0x69,0x00,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_mov_b32 v7, vcc_lo
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_mov_b32 v7, vcc_lo ; encoding: [0x7b,0x80,0x20,0xcf,0x6a,0x00,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_mov_b32 v7, vcc_hi
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_mov_b32 v7, vcc_hi ; encoding: [0x7f,0x80,0x20,0xcf,0x6b,0x00,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_mov_b32 v7, ttmp15
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_mov_b32 v7, ttmp15 ; encoding: [0x7e,0x80,0x20,0xcf,0x7b,0x00,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_mov_b32 v7, m0
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_mov_b32 v7, m0 ; encoding: [0x7d,0x80,0x20,0xcf,0x7d,0x00,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_mov_b32 v7, exec_lo
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_mov_b32 v7, exec_lo ; encoding: [0x6b,0x80,0x20,0xcf,0x7e,0x00,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_mov_b32 v7, exec_hi
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_mov_b32 v7, exec_hi ; encoding: [0x6a,0x80,0x20,0xcf,0x7f,0x00,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_mov_b32 v7, -1
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_mov_b32 v7, -1 ; encoding: [0xfd,0x80,0x20,0xcf,0xc1,0x00,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_mov_b32 v7, 0.5
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_mov_b32 v7, 0.5 ; encoding: [0xf0,0x80,0x20,0xcf,0xf0,0x00,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_mov_b32 v7, src_scc
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_mov_b32 v7, src_scc ; encoding: [0xc1,0x80,0x20,0xcf,0xfd,0x00,0x00,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v25, v8 :: v_dual_mov_b32 v13, v16
// GFX1250: v_dual_mov_b32 v25, v8 :: v_dual_mov_b32 v13, v16 ; encoding: [0x08,0x81,0x20,0xcf,0x10,0x01,0x00,0x00,0x19,0x00,0x00,0x0d]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_mul_dx9_zero_f32 v7, v1, v255
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_mul_dx9_zero_f32 v7, v1, v255 ; encoding: [0x04,0x71,0x20,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_mul_dx9_zero_f32 v7, v255, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_mul_dx9_zero_f32 v7, v255, v255 ; encoding: [0x01,0x71,0x20,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_mul_dx9_zero_f32 v7, v2, v255
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_mul_dx9_zero_f32 v7, v2, v255 ; encoding: [0xff,0x71,0x20,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v255
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v255 ; encoding: [0x02,0x71,0x20,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_mul_dx9_zero_f32 v7, v4, v255
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_mul_dx9_zero_f32 v7, v4, v255 ; encoding: [0x03,0x71,0x20,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_mul_dx9_zero_f32 v7, s1, v255
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_mul_dx9_zero_f32 v7, s1, v255 ; encoding: [0x69,0x70,0x20,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_mul_dx9_zero_f32 v7, s105, v255
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_mul_dx9_zero_f32 v7, s105, v255 ; encoding: [0x01,0x70,0x20,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v255
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v255 ; encoding: [0x7b,0x70,0x20,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v255
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v255 ; encoding: [0x7f,0x70,0x20,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v255
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v255 ; encoding: [0x7e,0x70,0x20,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_mul_dx9_zero_f32 v7, m0, v255
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_mul_dx9_zero_f32 v7, m0, v255 ; encoding: [0x7d,0x70,0x20,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v255
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v255 ; encoding: [0x6b,0x70,0x20,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v255
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v255 ; encoding: [0x6a,0x70,0x20,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_mul_dx9_zero_f32 v7, -1, v255
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_mul_dx9_zero_f32 v7, -1, v255 ; encoding: [0xfd,0x70,0x20,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v3
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v3 ; encoding: [0xf0,0x70,0x20,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v4
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v4 ; encoding: [0xc1,0x70,0x20,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_mul_f32 v7, v1, v255
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_mul_f32 v7, v1, v255 ; encoding: [0x04,0x31,0x20,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_mul_f32 v7, v255, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_mul_f32 v7, v255, v255 ; encoding: [0x01,0x31,0x20,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_mul_f32 v7, v2, v255
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_mul_f32 v7, v2, v255 ; encoding: [0xff,0x31,0x20,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_mul_f32 v7, v3, v255
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_mul_f32 v7, v3, v255 ; encoding: [0x02,0x31,0x20,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_mul_f32 v7, v4, v255
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_mul_f32 v7, v4, v255 ; encoding: [0x03,0x31,0x20,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_mul_f32 v7, s1, v255
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_mul_f32 v7, s1, v255 ; encoding: [0x69,0x30,0x20,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_mul_f32 v7, s105, v255
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_mul_f32 v7, s105, v255 ; encoding: [0x01,0x30,0x20,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_mul_f32 v7, vcc_lo, v255
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_mul_f32 v7, vcc_lo, v255 ; encoding: [0x7b,0x30,0x20,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_mul_f32 v7, vcc_hi, v255
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_mul_f32 v7, vcc_hi, v255 ; encoding: [0x7f,0x30,0x20,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_mul_f32 v7, ttmp15, v255
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_mul_f32 v7, ttmp15, v255 ; encoding: [0x7e,0x30,0x20,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_mul_f32 v7, m0, v255
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_mul_f32 v7, m0, v255 ; encoding: [0x7d,0x30,0x20,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_mul_f32 v7, exec_lo, v255
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_mul_f32 v7, exec_lo, v255 ; encoding: [0x6b,0x30,0x20,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_mul_f32 v7, exec_hi, v255
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_mul_f32 v7, exec_hi, v255 ; encoding: [0x6a,0x30,0x20,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_mul_f32 v7, -1, v255
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_mul_f32 v7, -1, v255 ; encoding: [0xfd,0x30,0x20,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_mul_f32 v7, 0.5, v3
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_mul_f32 v7, 0.5, v3 ; encoding: [0xf0,0x30,0x20,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_mul_f32 v7, src_scc, v4
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_mul_f32 v7, src_scc, v4 ; encoding: [0xc1,0x30,0x20,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_sub_f32 v7, v1, v255
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_sub_f32 v7, v1, v255 ; encoding: [0x04,0x51,0x20,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_sub_f32 v7, v255, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_sub_f32 v7, v255, v255 ; encoding: [0x01,0x51,0x20,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_sub_f32 v7, v2, v255
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_sub_f32 v7, v2, v255 ; encoding: [0xff,0x51,0x20,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_sub_f32 v7, v3, v255
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_sub_f32 v7, v3, v255 ; encoding: [0x02,0x51,0x20,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_sub_f32 v7, v4, v255
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_sub_f32 v7, v4, v255 ; encoding: [0x03,0x51,0x20,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_sub_f32 v7, s1, v255
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_sub_f32 v7, s1, v255 ; encoding: [0x69,0x50,0x20,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_sub_f32 v7, s105, v255
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_sub_f32 v7, s105, v255 ; encoding: [0x01,0x50,0x20,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_sub_f32 v7, vcc_lo, v255
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_sub_f32 v7, vcc_lo, v255 ; encoding: [0x7b,0x50,0x20,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_sub_f32 v7, vcc_hi, v255
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_sub_f32 v7, vcc_hi, v255 ; encoding: [0x7f,0x50,0x20,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_sub_f32 v7, ttmp15, v255
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_sub_f32 v7, ttmp15, v255 ; encoding: [0x7e,0x50,0x20,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_sub_f32 v7, m0, v255
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_sub_f32 v7, m0, v255 ; encoding: [0x7d,0x50,0x20,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_sub_f32 v7, exec_lo, v255
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_sub_f32 v7, exec_lo, v255 ; encoding: [0x6b,0x50,0x20,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_sub_f32 v7, exec_hi, v255
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_sub_f32 v7, exec_hi, v255 ; encoding: [0x6a,0x50,0x20,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_sub_f32 v7, -1, v255
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_sub_f32 v7, -1, v255 ; encoding: [0xfd,0x50,0x20,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_sub_f32 v7, 0.5, v3
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_sub_f32 v7, 0.5, v3 ; encoding: [0xf0,0x50,0x20,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_sub_f32 v7, src_scc, v4
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_sub_f32 v7, src_scc, v4 ; encoding: [0xc1,0x50,0x20,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_subrev_f32 v7, v1, v255
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_subrev_f32 v7, v1, v255 ; encoding: [0x04,0x61,0x20,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_subrev_f32 v7, v255, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_subrev_f32 v7, v255, v255 ; encoding: [0x01,0x61,0x20,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_subrev_f32 v7, v2, v255
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_subrev_f32 v7, v2, v255 ; encoding: [0xff,0x61,0x20,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_subrev_f32 v7, v3, v255
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_subrev_f32 v7, v3, v255 ; encoding: [0x02,0x61,0x20,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_subrev_f32 v7, v4, v255
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_subrev_f32 v7, v4, v255 ; encoding: [0x03,0x61,0x20,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_subrev_f32 v7, s1, v255
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_subrev_f32 v7, s1, v255 ; encoding: [0x69,0x60,0x20,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_subrev_f32 v7, s105, v255
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_subrev_f32 v7, s105, v255 ; encoding: [0x01,0x60,0x20,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_subrev_f32 v7, vcc_lo, v255
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_subrev_f32 v7, vcc_lo, v255 ; encoding: [0x7b,0x60,0x20,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_subrev_f32 v7, vcc_hi, v255
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_subrev_f32 v7, vcc_hi, v255 ; encoding: [0x7f,0x60,0x20,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_subrev_f32 v7, ttmp15, v255
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_subrev_f32 v7, ttmp15, v255 ; encoding: [0x7e,0x60,0x20,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_subrev_f32 v7, m0, v255
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_subrev_f32 v7, m0, v255 ; encoding: [0x7d,0x60,0x20,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_subrev_f32 v7, exec_lo, v255
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_subrev_f32 v7, exec_lo, v255 ; encoding: [0x6b,0x60,0x20,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_subrev_f32 v7, exec_hi, v255
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_subrev_f32 v7, exec_hi, v255 ; encoding: [0x6a,0x60,0x20,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_subrev_f32 v7, -1, v255
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_subrev_f32 v7, -1, v255 ; encoding: [0xfd,0x60,0x20,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_subrev_f32 v7, 0.5, v3
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_subrev_f32 v7, 0.5, v3 ; encoding: [0xf0,0x60,0x20,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_subrev_f32 v7, src_scc, v4
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_subrev_f32 v7, src_scc, v4 ; encoding: [0xc1,0x60,0x20,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_fma_f32 v7, v1, v3, v4
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_fma_f32 v7, v1, v3, v4 ; encoding: [0x04,0x31,0x21,0xcf,0x01,0x01,0x00,0x00,0xff,0x03,0x04,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:254
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:0xfe ; encoding: [0x04,0x21,0x21,0xcf,0x01,0x01,0x00,0x00,0xff,0x03,0xfe,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3 ; encoding: [0x04,0x41,0x1c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3 ; encoding: [0x01,0x41,0x1c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3 ; encoding: [0xff,0x41,0x1c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3 ; encoding: [0x02,0x41,0x1c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3 ; encoding: [0x03,0x41,0x1c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3 ; encoding: [0x69,0x40,0x1c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3 ; encoding: [0x01,0x40,0x1c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x1c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x1c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x1c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3 ; encoding: [0x7d,0x40,0x1c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x1c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x1c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3 ; encoding: [0xfd,0x40,0x1c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x1c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x1c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3 ; encoding: [0x04,0x01,0x1d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3 ; encoding: [0x01,0x01,0x1d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3 ; encoding: [0xff,0x01,0x1d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3 ; encoding: [0x02,0x01,0x1d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3 ; encoding: [0x03,0x01,0x1d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3 ; encoding: [0x69,0x00,0x1d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3 ; encoding: [0x01,0x00,0x1d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x1d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x1d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x1d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x00,0x1d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x1d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x1d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x00,0x1d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x1d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x1d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo ; encoding: [0x04,0x91,0x1c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo ; encoding: [0x01,0x91,0x1c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo ; encoding: [0xff,0x91,0x1c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo ; encoding: [0x02,0x91,0x1c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo ; encoding: [0x03,0x91,0x1c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo ; encoding: [0x69,0x90,0x1c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo ; encoding: [0x01,0x90,0x1c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo ; encoding: [0x7b,0x90,0x1c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo ; encoding: [0x7f,0x90,0x1c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo ; encoding: [0x7e,0x90,0x1c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo ; encoding: [0x7d,0x90,0x1c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo ; encoding: [0x6b,0x90,0x1c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo ; encoding: [0x6a,0x90,0x1c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo ; encoding: [0xfd,0x90,0x1c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo ; encoding: [0xf0,0x90,0x1c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo ; encoding: [0xc1,0x90,0x1c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3 ; encoding: [0x04,0x01,0x1c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3 ; encoding: [0x01,0x01,0x1c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3 ; encoding: [0xff,0x01,0x1c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3 ; encoding: [0x02,0x01,0x1c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3 ; encoding: [0x03,0x01,0x1c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3 ; encoding: [0x69,0x00,0x1c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3 ; encoding: [0x01,0x00,0x1c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x1c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x1c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x1c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3 ; encoding: [0x7d,0x00,0x1c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x1c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x1c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3 ; encoding: [0xfd,0x00,0x1c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x1c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x1c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3 ; encoding: [0x04,0x11,0x1d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3 ; encoding: [0x01,0x11,0x1d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3 ; encoding: [0xff,0x11,0x1d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3 ; encoding: [0x02,0x11,0x1d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3 ; encoding: [0x03,0x11,0x1d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3 ; encoding: [0x69,0x10,0x1d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3 ; encoding: [0x01,0x10,0x1d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3 ; encoding: [0x7b,0x10,0x1d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3 ; encoding: [0x7f,0x10,0x1d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3 ; encoding: [0x7e,0x10,0x1d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3 ; encoding: [0x7d,0x10,0x1d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3 ; encoding: [0x6b,0x10,0x1d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3 ; encoding: [0x6a,0x10,0x1d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3 ; encoding: [0xfd,0x10,0x1d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2 ; encoding: [0xf0,0x10,0x1d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5 ; encoding: [0xc1,0x10,0x1d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3 ; encoding: [0x04,0xa1,0x1c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3 ; encoding: [0x01,0xa1,0x1c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3 ; encoding: [0xff,0xa1,0x1c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3 ; encoding: [0x02,0xa1,0x1c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3 ; encoding: [0x03,0xa1,0x1c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3 ; encoding: [0x69,0xa0,0x1c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3 ; encoding: [0x01,0xa0,0x1c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xa0,0x1c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xa0,0x1c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xa0,0x1c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3 ; encoding: [0x7d,0xa0,0x1c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xa0,0x1c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xa0,0x1c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3 ; encoding: [0xfd,0xa0,0x1c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xa0,0x1c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xa0,0x1c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3 ; encoding: [0x04,0xb1,0x1c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3 ; encoding: [0x01,0xb1,0x1c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3 ; encoding: [0xff,0xb1,0x1c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3 ; encoding: [0x02,0xb1,0x1c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3 ; encoding: [0x03,0xb1,0x1c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3 ; encoding: [0x69,0xb0,0x1c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3 ; encoding: [0x01,0xb0,0x1c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xb0,0x1c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xb0,0x1c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xb0,0x1c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3 ; encoding: [0x7d,0xb0,0x1c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xb0,0x1c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xb0,0x1c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3 ; encoding: [0xfd,0xb0,0x1c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xb0,0x1c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xb0,0x1c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1 ; encoding: [0x04,0x81,0x1c,0xcf,0x01,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255 ; encoding: [0x01,0x81,0x1c,0xcf,0xff,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2 ; encoding: [0xff,0x81,0x1c,0xcf,0x02,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3 ; encoding: [0x02,0x81,0x1c,0xcf,0x03,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4 ; encoding: [0x03,0x81,0x1c,0xcf,0x04,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1 ; encoding: [0x69,0x80,0x1c,0xcf,0x01,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105 ; encoding: [0x01,0x80,0x1c,0xcf,0x69,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo ; encoding: [0x7b,0x80,0x1c,0xcf,0x6a,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi ; encoding: [0x7f,0x80,0x1c,0xcf,0x6b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15 ; encoding: [0x7e,0x80,0x1c,0xcf,0x7b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0 ; encoding: [0x7d,0x80,0x1c,0xcf,0x7d,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo ; encoding: [0x6b,0x80,0x1c,0xcf,0x7e,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi ; encoding: [0x6a,0x80,0x1c,0xcf,0x7f,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1 ; encoding: [0xfd,0x80,0x1c,0xcf,0xc1,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5 ; encoding: [0xf0,0x80,0x1c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc ; encoding: [0xc1,0x80,0x1c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3 ; encoding: [0x04,0x71,0x1c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3 ; encoding: [0x01,0x71,0x1c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3 ; encoding: [0xff,0x71,0x1c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3 ; encoding: [0x02,0x71,0x1c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3 ; encoding: [0x03,0x71,0x1c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3 ; encoding: [0x69,0x70,0x1c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3 ; encoding: [0x01,0x70,0x1c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x1c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x1c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x1c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3 ; encoding: [0x7d,0x70,0x1c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x1c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x1c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3 ; encoding: [0xfd,0x70,0x1c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x1c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x1c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3 ; encoding: [0x04,0x31,0x1c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3 ; encoding: [0x01,0x31,0x1c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3 ; encoding: [0xff,0x31,0x1c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3 ; encoding: [0x02,0x31,0x1c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3 ; encoding: [0x03,0x31,0x1c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3 ; encoding: [0x69,0x30,0x1c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3 ; encoding: [0x01,0x30,0x1c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x30,0x1c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x30,0x1c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x30,0x1c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3 ; encoding: [0x7d,0x30,0x1c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x30,0x1c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x30,0x1c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3 ; encoding: [0xfd,0x30,0x1c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2 ; encoding: [0xf0,0x30,0x1c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5 ; encoding: [0xc1,0x30,0x1c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3 ; encoding: [0x04,0x51,0x1c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3 ; encoding: [0x01,0x51,0x1c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3 ; encoding: [0xff,0x51,0x1c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3 ; encoding: [0x02,0x51,0x1c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3 ; encoding: [0x03,0x51,0x1c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3 ; encoding: [0x69,0x50,0x1c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3 ; encoding: [0x01,0x50,0x1c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x50,0x1c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x50,0x1c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x50,0x1c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3 ; encoding: [0x7d,0x50,0x1c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x50,0x1c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x50,0x1c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3 ; encoding: [0xfd,0x50,0x1c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2 ; encoding: [0xf0,0x50,0x1c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5 ; encoding: [0xc1,0x50,0x1c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3 ; encoding: [0x04,0x61,0x1c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3 ; encoding: [0x01,0x61,0x1c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3 ; encoding: [0xff,0x61,0x1c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3 ; encoding: [0x02,0x61,0x1c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3 ; encoding: [0x03,0x61,0x1c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3 ; encoding: [0x69,0x60,0x1c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3 ; encoding: [0x01,0x60,0x1c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x60,0x1c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x60,0x1c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x60,0x1c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3 ; encoding: [0x7d,0x60,0x1c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x60,0x1c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x60,0x1c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3 ; encoding: [0xfd,0x60,0x1c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2 ; encoding: [0xf0,0x60,0x1c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5 ; encoding: [0xc1,0x60,0x1c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4 ; encoding: [0x04,0x31,0x1d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x04,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:0x11
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:0x11 ; encoding: [0x04,0x21,0x1d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x11,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3 ; encoding: [0x04,0x41,0x0c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3 ; encoding: [0x01,0x41,0x0c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3 ; encoding: [0xff,0x41,0x0c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3 ; encoding: [0x02,0x41,0x0c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3 ; encoding: [0x03,0x41,0x0c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3 ; encoding: [0x69,0x40,0x0c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3 ; encoding: [0x01,0x40,0x0c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x0c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x0c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x0c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3 ; encoding: [0x7d,0x40,0x0c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x0c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x0c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3 ; encoding: [0xfd,0x40,0x0c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x0c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x0c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3 ; encoding: [0x04,0x01,0x0d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3 ; encoding: [0x01,0x01,0x0d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3 ; encoding: [0xff,0x01,0x0d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3 ; encoding: [0x02,0x01,0x0d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3 ; encoding: [0x03,0x01,0x0d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3 ; encoding: [0x69,0x00,0x0d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3 ; encoding: [0x01,0x00,0x0d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x0d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x0d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x0d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x00,0x0d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x0d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x0d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x00,0x0d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x0d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x0d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo ; encoding: [0x04,0x91,0x0c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo ; encoding: [0x01,0x91,0x0c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo ; encoding: [0xff,0x91,0x0c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo ; encoding: [0x02,0x91,0x0c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo ; encoding: [0x03,0x91,0x0c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo ; encoding: [0x69,0x90,0x0c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo ; encoding: [0x01,0x90,0x0c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo ; encoding: [0x7b,0x90,0x0c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo ; encoding: [0x7f,0x90,0x0c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo ; encoding: [0x7e,0x90,0x0c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo ; encoding: [0x7d,0x90,0x0c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo ; encoding: [0x6b,0x90,0x0c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo ; encoding: [0x6a,0x90,0x0c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo ; encoding: [0xfd,0x90,0x0c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo ; encoding: [0xf0,0x90,0x0c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo ; encoding: [0xc1,0x90,0x0c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3 ; encoding: [0x04,0x01,0x0c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3 ; encoding: [0x01,0x01,0x0c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3 ; encoding: [0xff,0x01,0x0c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3 ; encoding: [0x02,0x01,0x0c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3 ; encoding: [0x03,0x01,0x0c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3 ; encoding: [0x69,0x00,0x0c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3 ; encoding: [0x01,0x00,0x0c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x0c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x0c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x0c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3 ; encoding: [0x7d,0x00,0x0c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x0c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x0c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3 ; encoding: [0xfd,0x00,0x0c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x0c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x0c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3 ; encoding: [0x04,0x11,0x0d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3 ; encoding: [0x01,0x11,0x0d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3 ; encoding: [0xff,0x11,0x0d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3 ; encoding: [0x02,0x11,0x0d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3 ; encoding: [0x03,0x11,0x0d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3 ; encoding: [0x69,0x10,0x0d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3 ; encoding: [0x01,0x10,0x0d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3 ; encoding: [0x7b,0x10,0x0d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3 ; encoding: [0x7f,0x10,0x0d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3 ; encoding: [0x7e,0x10,0x0d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3 ; encoding: [0x7d,0x10,0x0d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3 ; encoding: [0x6b,0x10,0x0d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3 ; encoding: [0x6a,0x10,0x0d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3 ; encoding: [0xfd,0x10,0x0d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2 ; encoding: [0xf0,0x10,0x0d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5 ; encoding: [0xc1,0x10,0x0d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3 ; encoding: [0x04,0xa1,0x0c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3 ; encoding: [0x01,0xa1,0x0c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3 ; encoding: [0xff,0xa1,0x0c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3 ; encoding: [0x02,0xa1,0x0c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3 ; encoding: [0x03,0xa1,0x0c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3 ; encoding: [0x69,0xa0,0x0c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3 ; encoding: [0x01,0xa0,0x0c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xa0,0x0c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xa0,0x0c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xa0,0x0c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3 ; encoding: [0x7d,0xa0,0x0c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xa0,0x0c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xa0,0x0c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3 ; encoding: [0xfd,0xa0,0x0c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xa0,0x0c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xa0,0x0c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3 ; encoding: [0x04,0xb1,0x0c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3 ; encoding: [0x01,0xb1,0x0c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3 ; encoding: [0xff,0xb1,0x0c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3 ; encoding: [0x02,0xb1,0x0c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3 ; encoding: [0x03,0xb1,0x0c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3 ; encoding: [0x69,0xb0,0x0c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3 ; encoding: [0x01,0xb0,0x0c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xb0,0x0c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xb0,0x0c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xb0,0x0c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3 ; encoding: [0x7d,0xb0,0x0c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xb0,0x0c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xb0,0x0c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3 ; encoding: [0xfd,0xb0,0x0c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xb0,0x0c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xb0,0x0c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1
// GFX1250: v_dual_mul_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1 ; encoding: [0x04,0x81,0x0c,0xcf,0x01,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255
// GFX1250: v_dual_mul_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255 ; encoding: [0x01,0x81,0x0c,0xcf,0xff,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2
// GFX1250: v_dual_mul_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2 ; encoding: [0xff,0x81,0x0c,0xcf,0x02,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3
// GFX1250: v_dual_mul_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3 ; encoding: [0x02,0x81,0x0c,0xcf,0x03,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4
// GFX1250: v_dual_mul_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4 ; encoding: [0x03,0x81,0x0c,0xcf,0x04,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1
// GFX1250: v_dual_mul_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1 ; encoding: [0x69,0x80,0x0c,0xcf,0x01,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105
// GFX1250: v_dual_mul_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105 ; encoding: [0x01,0x80,0x0c,0xcf,0x69,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo
// GFX1250: v_dual_mul_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo ; encoding: [0x7b,0x80,0x0c,0xcf,0x6a,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi
// GFX1250: v_dual_mul_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi ; encoding: [0x7f,0x80,0x0c,0xcf,0x6b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15
// GFX1250: v_dual_mul_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15 ; encoding: [0x7e,0x80,0x0c,0xcf,0x7b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0
// GFX1250: v_dual_mul_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0 ; encoding: [0x7d,0x80,0x0c,0xcf,0x7d,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo ; encoding: [0x6b,0x80,0x0c,0xcf,0x7e,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi ; encoding: [0x6a,0x80,0x0c,0xcf,0x7f,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1
// GFX1250: v_dual_mul_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1 ; encoding: [0xfd,0x80,0x0c,0xcf,0xc1,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5 ; encoding: [0xf0,0x80,0x0c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc ; encoding: [0xc1,0x80,0x0c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3 ; encoding: [0x04,0x71,0x0c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3 ; encoding: [0x01,0x71,0x0c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3 ; encoding: [0xff,0x71,0x0c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3 ; encoding: [0x02,0x71,0x0c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3 ; encoding: [0x03,0x71,0x0c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3 ; encoding: [0x69,0x70,0x0c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3 ; encoding: [0x01,0x70,0x0c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x0c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x0c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x0c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3 ; encoding: [0x7d,0x70,0x0c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x0c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x0c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3 ; encoding: [0xfd,0x70,0x0c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x0c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x0c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3 ; encoding: [0x04,0x31,0x0c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3 ; encoding: [0x01,0x31,0x0c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3 ; encoding: [0xff,0x31,0x0c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3 ; encoding: [0x02,0x31,0x0c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3 ; encoding: [0x03,0x31,0x0c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3 ; encoding: [0x69,0x30,0x0c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3 ; encoding: [0x01,0x30,0x0c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x30,0x0c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x30,0x0c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x30,0x0c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3 ; encoding: [0x7d,0x30,0x0c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x30,0x0c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x30,0x0c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3 ; encoding: [0xfd,0x30,0x0c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2 ; encoding: [0xf0,0x30,0x0c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5 ; encoding: [0xc1,0x30,0x0c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3 ; encoding: [0x04,0x51,0x0c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3 ; encoding: [0x01,0x51,0x0c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3 ; encoding: [0xff,0x51,0x0c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3 ; encoding: [0x02,0x51,0x0c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3 ; encoding: [0x03,0x51,0x0c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3 ; encoding: [0x69,0x50,0x0c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3 ; encoding: [0x01,0x50,0x0c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x50,0x0c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x50,0x0c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x50,0x0c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3 ; encoding: [0x7d,0x50,0x0c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x50,0x0c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x50,0x0c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3 ; encoding: [0xfd,0x50,0x0c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2 ; encoding: [0xf0,0x50,0x0c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5 ; encoding: [0xc1,0x50,0x0c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3 ; encoding: [0x04,0x61,0x0c,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3 ; encoding: [0x01,0x61,0x0c,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3 ; encoding: [0xff,0x61,0x0c,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3 ; encoding: [0x02,0x61,0x0c,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3 ; encoding: [0x03,0x61,0x0c,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3 ; encoding: [0x69,0x60,0x0c,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3 ; encoding: [0x01,0x60,0x0c,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x60,0x0c,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x60,0x0c,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x60,0x0c,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3 ; encoding: [0x7d,0x60,0x0c,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x60,0x0c,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x60,0x0c,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3 ; encoding: [0xfd,0x60,0x0c,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2 ; encoding: [0xf0,0x60,0x0c,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5 ; encoding: [0xc1,0x60,0x0c,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4 ; encoding: [0x04,0x31,0x0d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x04,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:0x71
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:0x71 ; encoding: [0x04,0x21,0x0d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x71,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3 ; encoding: [0x04,0x41,0x14,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3 ; encoding: [0x01,0x41,0x14,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3 ; encoding: [0xff,0x41,0x14,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3 ; encoding: [0x02,0x41,0x14,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3 ; encoding: [0x03,0x41,0x14,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3 ; encoding: [0x69,0x40,0x14,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3 ; encoding: [0x01,0x40,0x14,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x14,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x14,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x14,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3 ; encoding: [0x7d,0x40,0x14,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x14,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x14,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3 ; encoding: [0xfd,0x40,0x14,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x14,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x14,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3 ; encoding: [0x04,0x01,0x15,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3 ; encoding: [0x01,0x01,0x15,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3 ; encoding: [0xff,0x01,0x15,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3 ; encoding: [0x02,0x01,0x15,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3 ; encoding: [0x03,0x01,0x15,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3 ; encoding: [0x69,0x00,0x15,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3 ; encoding: [0x01,0x00,0x15,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x15,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x15,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x15,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x00,0x15,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x15,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x15,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x00,0x15,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x15,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x15,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo ; encoding: [0x04,0x91,0x14,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo ; encoding: [0x01,0x91,0x14,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo ; encoding: [0xff,0x91,0x14,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo ; encoding: [0x02,0x91,0x14,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo ; encoding: [0x03,0x91,0x14,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo ; encoding: [0x69,0x90,0x14,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo ; encoding: [0x01,0x90,0x14,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo ; encoding: [0x7b,0x90,0x14,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo ; encoding: [0x7f,0x90,0x14,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo ; encoding: [0x7e,0x90,0x14,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo ; encoding: [0x7d,0x90,0x14,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo ; encoding: [0x6b,0x90,0x14,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo ; encoding: [0x6a,0x90,0x14,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo ; encoding: [0xfd,0x90,0x14,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo ; encoding: [0xf0,0x90,0x14,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo ; encoding: [0xc1,0x90,0x14,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3 ; encoding: [0x04,0x01,0x14,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3 ; encoding: [0x01,0x01,0x14,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3 ; encoding: [0xff,0x01,0x14,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3 ; encoding: [0x02,0x01,0x14,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3 ; encoding: [0x03,0x01,0x14,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3 ; encoding: [0x69,0x00,0x14,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3 ; encoding: [0x01,0x00,0x14,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x14,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x14,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x14,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3 ; encoding: [0x7d,0x00,0x14,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x14,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x14,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3 ; encoding: [0xfd,0x00,0x14,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x14,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x14,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3 ; encoding: [0x04,0x11,0x15,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3 ; encoding: [0x01,0x11,0x15,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3 ; encoding: [0xff,0x11,0x15,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3 ; encoding: [0x02,0x11,0x15,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3 ; encoding: [0x03,0x11,0x15,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3 ; encoding: [0x69,0x10,0x15,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3 ; encoding: [0x01,0x10,0x15,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3 ; encoding: [0x7b,0x10,0x15,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3 ; encoding: [0x7f,0x10,0x15,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3 ; encoding: [0x7e,0x10,0x15,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3 ; encoding: [0x7d,0x10,0x15,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3 ; encoding: [0x6b,0x10,0x15,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3 ; encoding: [0x6a,0x10,0x15,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3 ; encoding: [0xfd,0x10,0x15,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2 ; encoding: [0xf0,0x10,0x15,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5 ; encoding: [0xc1,0x10,0x15,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3 ; encoding: [0x04,0xa1,0x14,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3 ; encoding: [0x01,0xa1,0x14,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3 ; encoding: [0xff,0xa1,0x14,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3 ; encoding: [0x02,0xa1,0x14,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3 ; encoding: [0x03,0xa1,0x14,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3 ; encoding: [0x69,0xa0,0x14,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3 ; encoding: [0x01,0xa0,0x14,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xa0,0x14,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xa0,0x14,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xa0,0x14,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3 ; encoding: [0x7d,0xa0,0x14,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xa0,0x14,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xa0,0x14,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3 ; encoding: [0xfd,0xa0,0x14,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xa0,0x14,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xa0,0x14,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3 ; encoding: [0x04,0xb1,0x14,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3 ; encoding: [0x01,0xb1,0x14,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3 ; encoding: [0xff,0xb1,0x14,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3 ; encoding: [0x02,0xb1,0x14,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3 ; encoding: [0x03,0xb1,0x14,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3 ; encoding: [0x69,0xb0,0x14,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3 ; encoding: [0x01,0xb0,0x14,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xb0,0x14,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xb0,0x14,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xb0,0x14,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3 ; encoding: [0x7d,0xb0,0x14,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xb0,0x14,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xb0,0x14,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3 ; encoding: [0xfd,0xb0,0x14,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xb0,0x14,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xb0,0x14,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1
// GFX1250: v_dual_sub_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1 ; encoding: [0x04,0x81,0x14,0xcf,0x01,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255
// GFX1250: v_dual_sub_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255 ; encoding: [0x01,0x81,0x14,0xcf,0xff,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2
// GFX1250: v_dual_sub_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2 ; encoding: [0xff,0x81,0x14,0xcf,0x02,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3
// GFX1250: v_dual_sub_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3 ; encoding: [0x02,0x81,0x14,0xcf,0x03,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4
// GFX1250: v_dual_sub_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4 ; encoding: [0x03,0x81,0x14,0xcf,0x04,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1
// GFX1250: v_dual_sub_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1 ; encoding: [0x69,0x80,0x14,0xcf,0x01,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105
// GFX1250: v_dual_sub_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105 ; encoding: [0x01,0x80,0x14,0xcf,0x69,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo
// GFX1250: v_dual_sub_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo ; encoding: [0x7b,0x80,0x14,0xcf,0x6a,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi
// GFX1250: v_dual_sub_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi ; encoding: [0x7f,0x80,0x14,0xcf,0x6b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15
// GFX1250: v_dual_sub_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15 ; encoding: [0x7e,0x80,0x14,0xcf,0x7b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0
// GFX1250: v_dual_sub_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0 ; encoding: [0x7d,0x80,0x14,0xcf,0x7d,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo ; encoding: [0x6b,0x80,0x14,0xcf,0x7e,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi ; encoding: [0x6a,0x80,0x14,0xcf,0x7f,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1
// GFX1250: v_dual_sub_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1 ; encoding: [0xfd,0x80,0x14,0xcf,0xc1,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5 ; encoding: [0xf0,0x80,0x14,0xcf,0xf0,0x00,0x03,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc ; encoding: [0xc1,0x80,0x14,0xcf,0xfd,0x00,0x04,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3 ; encoding: [0x04,0x71,0x14,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3 ; encoding: [0x01,0x71,0x14,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3 ; encoding: [0xff,0x71,0x14,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3 ; encoding: [0x02,0x71,0x14,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3 ; encoding: [0x03,0x71,0x14,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3 ; encoding: [0x69,0x70,0x14,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3 ; encoding: [0x01,0x70,0x14,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x14,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x14,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x14,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3 ; encoding: [0x7d,0x70,0x14,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x14,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x14,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3 ; encoding: [0xfd,0x70,0x14,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x14,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x14,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3 ; encoding: [0x04,0x31,0x14,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3 ; encoding: [0x01,0x31,0x14,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3 ; encoding: [0xff,0x31,0x14,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3 ; encoding: [0x02,0x31,0x14,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3 ; encoding: [0x03,0x31,0x14,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3 ; encoding: [0x69,0x30,0x14,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3 ; encoding: [0x01,0x30,0x14,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x30,0x14,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x30,0x14,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x30,0x14,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3 ; encoding: [0x7d,0x30,0x14,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x30,0x14,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x30,0x14,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3 ; encoding: [0xfd,0x30,0x14,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2 ; encoding: [0xf0,0x30,0x14,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5 ; encoding: [0xc1,0x30,0x14,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3 ; encoding: [0x04,0x51,0x14,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3 ; encoding: [0x01,0x51,0x14,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3 ; encoding: [0xff,0x51,0x14,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3 ; encoding: [0x02,0x51,0x14,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3 ; encoding: [0x03,0x51,0x14,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3 ; encoding: [0x69,0x50,0x14,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3 ; encoding: [0x01,0x50,0x14,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x50,0x14,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x50,0x14,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x50,0x14,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3 ; encoding: [0x7d,0x50,0x14,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x50,0x14,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x50,0x14,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3 ; encoding: [0xfd,0x50,0x14,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2 ; encoding: [0xf0,0x50,0x14,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5 ; encoding: [0xc1,0x50,0x14,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3 ; encoding: [0x04,0x61,0x14,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3 ; encoding: [0x01,0x61,0x14,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3 ; encoding: [0xff,0x61,0x14,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3 ; encoding: [0x02,0x61,0x14,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3 ; encoding: [0x03,0x61,0x14,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3 ; encoding: [0x69,0x60,0x14,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3 ; encoding: [0x01,0x60,0x14,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x60,0x14,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x60,0x14,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x60,0x14,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3 ; encoding: [0x7d,0x60,0x14,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x60,0x14,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x60,0x14,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3 ; encoding: [0xfd,0x60,0x14,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2 ; encoding: [0xf0,0x60,0x14,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5 ; encoding: [0xc1,0x60,0x14,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4 ; encoding: [0x04,0x31,0x15,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x04,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:0x82
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:0x82 ; encoding: [0x04,0x21,0x15,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x82,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_add_f32 v7, v1, v3 ; encoding: [0x04,0x41,0x18,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_add_f32 v7, v255, v3 ; encoding: [0x01,0x41,0x18,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_add_f32 v7, v2, v3 ; encoding: [0xff,0x41,0x18,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_add_f32 v7, v3, v3 ; encoding: [0x02,0x41,0x18,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_add_f32 v7, v4, v3 ; encoding: [0x03,0x41,0x18,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_add_f32 v7, s1, v3 ; encoding: [0x69,0x40,0x18,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_add_f32 v7, s105, v3 ; encoding: [0x01,0x40,0x18,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_add_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x18,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_add_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x18,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_add_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x18,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_add_f32 v7, m0, v3 ; encoding: [0x7d,0x40,0x18,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_add_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x18,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_add_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x18,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_add_f32 v7, -1, v3 ; encoding: [0xfd,0x40,0x18,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_add_f32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x18,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_add_f32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x18,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_add_nc_u32 v7, v1, v3 ; encoding: [0x04,0x01,0x19,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_add_nc_u32 v7, v255, v3 ; encoding: [0x01,0x01,0x19,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_add_nc_u32 v7, v2, v3 ; encoding: [0xff,0x01,0x19,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_add_nc_u32 v7, v3, v3 ; encoding: [0x02,0x01,0x19,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_add_nc_u32 v7, v4, v3 ; encoding: [0x03,0x01,0x19,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_add_nc_u32 v7, s1, v3 ; encoding: [0x69,0x00,0x19,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_add_nc_u32 v7, s105, v3 ; encoding: [0x01,0x00,0x19,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_add_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x19,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_add_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x19,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_add_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x19,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_add_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x00,0x19,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_add_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x19,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_add_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x19,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_add_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x00,0x19,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_add_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x19,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_add_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x19,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_cndmask_b32 v7, v1, v3, vcc_lo ; encoding: [0x04,0x91,0x18,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_cndmask_b32 v7, v255, v3, vcc_lo ; encoding: [0x01,0x91,0x18,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_cndmask_b32 v7, v2, v3, vcc_lo ; encoding: [0xff,0x91,0x18,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_cndmask_b32 v7, v3, v3, vcc_lo ; encoding: [0x02,0x91,0x18,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_cndmask_b32 v7, v4, v3, vcc_lo ; encoding: [0x03,0x91,0x18,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_cndmask_b32 v7, s105, v3, vcc_lo ; encoding: [0x69,0x90,0x18,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_cndmask_b32 v7, s1, v3, vcc_lo ; encoding: [0x01,0x90,0x18,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_cndmask_b32 v7, ttmp15, v3, vcc_lo ; encoding: [0x7b,0x90,0x18,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_cndmask_b32 v7, exec_hi, v3, vcc_lo ; encoding: [0x7f,0x90,0x18,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_cndmask_b32 v7, exec_lo, v3, vcc_lo ; encoding: [0x7e,0x90,0x18,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_cndmask_b32 v7, m0, v3, vcc_lo ; encoding: [0x7d,0x90,0x18,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_cndmask_b32 v7, vcc_hi, v3, vcc_lo ; encoding: [0x6b,0x90,0x18,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_cndmask_b32 v7, vcc_lo, v3, vcc_lo ; encoding: [0x6a,0x90,0x18,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_cndmask_b32 v7, -1, v3, vcc_lo ; encoding: [0xfd,0x90,0x18,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_cndmask_b32 v7, 0.5, v2, vcc_lo ; encoding: [0xf0,0x90,0x18,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_cndmask_b32 v7, src_scc, v5, vcc_lo ; encoding: [0xc1,0x90,0x18,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x6a,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_fmac_f32 v7, v1, v3 ; encoding: [0x04,0x01,0x18,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_fmac_f32 v7, v255, v3 ; encoding: [0x01,0x01,0x18,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_fmac_f32 v7, v2, v3 ; encoding: [0xff,0x01,0x18,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_fmac_f32 v7, v3, v3 ; encoding: [0x02,0x01,0x18,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_fmac_f32 v7, v4, v3 ; encoding: [0x03,0x01,0x18,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3 ; encoding: [0x69,0x00,0x18,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_fmac_f32 v7, s105, v3 ; encoding: [0x01,0x00,0x18,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_fmac_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x00,0x18,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_fmac_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x00,0x18,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_fmac_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x00,0x18,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_fmac_f32 v7, m0, v3 ; encoding: [0x7d,0x00,0x18,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_fmac_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x00,0x18,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_fmac_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x00,0x18,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_fmac_f32 v7, -1, v3 ; encoding: [0xfd,0x00,0x18,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_fmac_f32 v7, 0.5, v2 ; encoding: [0xf0,0x00,0x18,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_fmac_f32 v7, src_scc, v5 ; encoding: [0xc1,0x00,0x18,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_lshlrev_b32 v7, v1, v3 ; encoding: [0x04,0x11,0x19,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_lshlrev_b32 v7, v255, v3 ; encoding: [0x01,0x11,0x19,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_lshlrev_b32 v7, v2, v3 ; encoding: [0xff,0x11,0x19,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_lshlrev_b32 v7, v3, v3 ; encoding: [0x02,0x11,0x19,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_lshlrev_b32 v7, v4, v3 ; encoding: [0x03,0x11,0x19,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_lshlrev_b32 v7, s1, v3 ; encoding: [0x69,0x10,0x19,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_lshlrev_b32 v7, s105, v3 ; encoding: [0x01,0x10,0x19,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_lshlrev_b32 v7, vcc_lo, v3 ; encoding: [0x7b,0x10,0x19,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_lshlrev_b32 v7, vcc_hi, v3 ; encoding: [0x7f,0x10,0x19,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_lshlrev_b32 v7, ttmp15, v3 ; encoding: [0x7e,0x10,0x19,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_lshlrev_b32 v7, m0, v3 ; encoding: [0x7d,0x10,0x19,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_lshlrev_b32 v7, exec_lo, v3 ; encoding: [0x6b,0x10,0x19,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_lshlrev_b32 v7, exec_hi, v3 ; encoding: [0x6a,0x10,0x19,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_lshlrev_b32 v7, -1, v3 ; encoding: [0xfd,0x10,0x19,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_lshlrev_b32 v7, 0.5, v2 ; encoding: [0xf0,0x10,0x19,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_lshlrev_b32 v7, src_scc, v5 ; encoding: [0xc1,0x10,0x19,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_max_num_f32 v7, v1, v3 ; encoding: [0x04,0xa1,0x18,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_max_num_f32 v7, v255, v3 ; encoding: [0x01,0xa1,0x18,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_max_num_f32 v7, v2, v3 ; encoding: [0xff,0xa1,0x18,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_max_num_f32 v7, v3, v3 ; encoding: [0x02,0xa1,0x18,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_max_num_f32 v7, v4, v3 ; encoding: [0x03,0xa1,0x18,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_max_num_f32 v7, s1, v3 ; encoding: [0x69,0xa0,0x18,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_max_num_f32 v7, s105, v3 ; encoding: [0x01,0xa0,0x18,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_max_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xa0,0x18,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_max_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xa0,0x18,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_max_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xa0,0x18,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_max_num_f32 v7, m0, v3 ; encoding: [0x7d,0xa0,0x18,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_max_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xa0,0x18,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_max_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xa0,0x18,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_max_num_f32 v7, -1, v3 ; encoding: [0xfd,0xa0,0x18,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_max_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xa0,0x18,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_max_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xa0,0x18,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_min_num_f32 v7, v1, v3 ; encoding: [0x04,0xb1,0x18,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_min_num_f32 v7, v255, v3 ; encoding: [0x01,0xb1,0x18,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_min_num_f32 v7, v2, v3 ; encoding: [0xff,0xb1,0x18,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_min_num_f32 v7, v3, v3 ; encoding: [0x02,0xb1,0x18,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_min_num_f32 v7, v4, v3 ; encoding: [0x03,0xb1,0x18,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_min_num_f32 v7, s1, v3 ; encoding: [0x69,0xb0,0x18,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_min_num_f32 v7, s105, v3 ; encoding: [0x01,0xb0,0x18,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_min_num_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0xb0,0x18,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_min_num_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0xb0,0x18,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_min_num_f32 v7, ttmp15, v3 ; encoding: [0x7e,0xb0,0x18,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_min_num_f32 v7, m0, v3 ; encoding: [0x7d,0xb0,0x18,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_min_num_f32 v7, exec_lo, v3 ; encoding: [0x6b,0xb0,0x18,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_min_num_f32 v7, exec_hi, v3 ; encoding: [0x6a,0xb0,0x18,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_min_num_f32 v7, -1, v3 ; encoding: [0xfd,0xb0,0x18,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_min_num_f32 v7, 0.5, v2 ; encoding: [0xf0,0xb0,0x18,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_min_num_f32 v7, src_scc, v5 ; encoding: [0xc1,0xb0,0x18,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1
// GFX1250: v_dual_subrev_f32 v255, v4, v255 :: v_dual_mov_b32 v7, v1 ; encoding: [0x04,0x81,0x18,0xcf,0x01,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255
// GFX1250: v_dual_subrev_f32 v255, v1, v255 :: v_dual_mov_b32 v7, v255 ; encoding: [0x01,0x81,0x18,0xcf,0xff,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2
// GFX1250: v_dual_subrev_f32 v255, v255, v255 :: v_dual_mov_b32 v7, v2 ; encoding: [0xff,0x81,0x18,0xcf,0x02,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v255 :: v_dual_mov_b32 v7, v3 ; encoding: [0x02,0x81,0x18,0xcf,0x03,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4
// GFX1250: v_dual_subrev_f32 v255, v3, v255 :: v_dual_mov_b32 v7, v4 ; encoding: [0x03,0x81,0x18,0xcf,0x04,0x01,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1
// GFX1250: v_dual_subrev_f32 v255, s105, v255 :: v_dual_mov_b32 v7, s1 ; encoding: [0x69,0x80,0x18,0xcf,0x01,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105
// GFX1250: v_dual_subrev_f32 v255, s1, v255 :: v_dual_mov_b32 v7, s105 ; encoding: [0x01,0x80,0x18,0xcf,0x69,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v255 :: v_dual_mov_b32 v7, vcc_lo ; encoding: [0x7b,0x80,0x18,0xcf,0x6a,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v255 :: v_dual_mov_b32 v7, vcc_hi ; encoding: [0x7f,0x80,0x18,0xcf,0x6b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v255 :: v_dual_mov_b32 v7, ttmp15 ; encoding: [0x7e,0x80,0x18,0xcf,0x7b,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0
// GFX1250: v_dual_subrev_f32 v255, m0, v255 :: v_dual_mov_b32 v7, m0 ; encoding: [0x7d,0x80,0x18,0xcf,0x7d,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v255 :: v_dual_mov_b32 v7, exec_lo ; encoding: [0x6b,0x80,0x18,0xcf,0x7e,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v255 :: v_dual_mov_b32 v7, exec_hi ; encoding: [0x6a,0x80,0x18,0xcf,0x7f,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1
// GFX1250: v_dual_subrev_f32 v255, src_scc, v255 :: v_dual_mov_b32 v7, -1 ; encoding: [0xfd,0x80,0x18,0xcf,0xc1,0x00,0xff,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_mov_b32 v7, 0.5 ; encoding: [0xf0,0x80,0x18,0xcf,0xf0,0x00,0x03,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_mov_b32 v7, src_scc ; encoding: [0xc1,0x80,0x18,0xcf,0xfd,0x00,0x04,0x00,0xff,0x00,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_mul_dx9_zero_f32 v7, v1, v3 ; encoding: [0x04,0x71,0x18,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_mul_dx9_zero_f32 v7, v255, v3 ; encoding: [0x01,0x71,0x18,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_mul_dx9_zero_f32 v7, v2, v3 ; encoding: [0xff,0x71,0x18,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_mul_dx9_zero_f32 v7, v3, v3 ; encoding: [0x02,0x71,0x18,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_mul_dx9_zero_f32 v7, v4, v3 ; encoding: [0x03,0x71,0x18,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_mul_dx9_zero_f32 v7, s1, v3 ; encoding: [0x69,0x70,0x18,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_mul_dx9_zero_f32 v7, s105, v3 ; encoding: [0x01,0x70,0x18,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x18,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x18,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x18,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_mul_dx9_zero_f32 v7, m0, v3 ; encoding: [0x7d,0x70,0x18,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x18,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_mul_dx9_zero_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x18,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_mul_dx9_zero_f32 v7, -1, v3 ; encoding: [0xfd,0x70,0x18,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_mul_dx9_zero_f32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x18,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_mul_dx9_zero_f32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x18,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_mul_f32 v7, v1, v3 ; encoding: [0x04,0x31,0x18,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_mul_f32 v7, v255, v3 ; encoding: [0x01,0x31,0x18,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_mul_f32 v7, v2, v3 ; encoding: [0xff,0x31,0x18,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_mul_f32 v7, v3, v3 ; encoding: [0x02,0x31,0x18,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_mul_f32 v7, v4, v3 ; encoding: [0x03,0x31,0x18,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_mul_f32 v7, s1, v3 ; encoding: [0x69,0x30,0x18,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_mul_f32 v7, s105, v3 ; encoding: [0x01,0x30,0x18,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_mul_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x30,0x18,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_mul_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x30,0x18,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_mul_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x30,0x18,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_mul_f32 v7, m0, v3 ; encoding: [0x7d,0x30,0x18,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_mul_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x30,0x18,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_mul_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x30,0x18,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_mul_f32 v7, -1, v3 ; encoding: [0xfd,0x30,0x18,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_mul_f32 v7, 0.5, v2 ; encoding: [0xf0,0x30,0x18,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_mul_f32 v7, src_scc, v5 ; encoding: [0xc1,0x30,0x18,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_sub_f32 v7, v1, v3 ; encoding: [0x04,0x51,0x18,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_sub_f32 v7, v255, v3 ; encoding: [0x01,0x51,0x18,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_sub_f32 v7, v2, v3 ; encoding: [0xff,0x51,0x18,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_sub_f32 v7, v3, v3 ; encoding: [0x02,0x51,0x18,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_sub_f32 v7, v4, v3 ; encoding: [0x03,0x51,0x18,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_sub_f32 v7, s1, v3 ; encoding: [0x69,0x50,0x18,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_sub_f32 v7, s105, v3 ; encoding: [0x01,0x50,0x18,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_sub_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x50,0x18,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_sub_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x50,0x18,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_sub_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x50,0x18,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_sub_f32 v7, m0, v3 ; encoding: [0x7d,0x50,0x18,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_sub_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x50,0x18,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_sub_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x50,0x18,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_sub_f32 v7, -1, v3 ; encoding: [0xfd,0x50,0x18,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_sub_f32 v7, 0.5, v2 ; encoding: [0xf0,0x50,0x18,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_sub_f32 v7, src_scc, v5 ; encoding: [0xc1,0x50,0x18,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_subrev_f32 v7, v1, v3 ; encoding: [0x04,0x61,0x18,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_subrev_f32 v7, v255, v3 ; encoding: [0x01,0x61,0x18,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_subrev_f32 v7, v2, v3 ; encoding: [0xff,0x61,0x18,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_subrev_f32 v7, v3, v3 ; encoding: [0x02,0x61,0x18,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_subrev_f32 v7, v4, v3 ; encoding: [0x03,0x61,0x18,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_subrev_f32 v7, s1, v3 ; encoding: [0x69,0x60,0x18,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_subrev_f32 v7, s105, v3 ; encoding: [0x01,0x60,0x18,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_subrev_f32 v7, vcc_lo, v3 ; encoding: [0x7b,0x60,0x18,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_subrev_f32 v7, vcc_hi, v3 ; encoding: [0x7f,0x60,0x18,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_subrev_f32 v7, ttmp15, v3 ; encoding: [0x7e,0x60,0x18,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_subrev_f32 v7, m0, v3 ; encoding: [0x7d,0x60,0x18,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_subrev_f32 v7, exec_lo, v3 ; encoding: [0x6b,0x60,0x18,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_subrev_f32 v7, exec_hi, v3 ; encoding: [0x6a,0x60,0x18,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_subrev_f32 v7, -1, v3 ; encoding: [0xfd,0x60,0x18,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_subrev_f32 v7, 0.5, v2 ; encoding: [0xf0,0x60,0x18,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_subrev_f32 v7, src_scc, v5 ; encoding: [0xc1,0x60,0x18,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_fma_f32 v7, v1, v3, v4 ; encoding: [0x04,0x31,0x19,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x04,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:0x83
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_bitop2_b32 v7, v1, v3 bitop3:0x83 ; encoding: [0x04,0x21,0x19,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x83,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3 ; encoding: [0x04,0x71,0x11,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3 ; encoding: [0x01,0x71,0x11,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3 ; encoding: [0xff,0x71,0x11,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3 ; encoding: [0x02,0x71,0x11,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3 ; encoding: [0x03,0x71,0x11,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3 ; encoding: [0x69,0x70,0x11,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3 ; encoding: [0x01,0x70,0x11,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x11,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x11,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x11,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3 ; encoding: [0x7d,0x70,0x11,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x11,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x11,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3 ; encoding: [0xfd,0x70,0x11,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x11,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x11,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_max_i32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_max_i32 v7, v1, v3 ; encoding: [0x04,0x71,0x25,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_max_i32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_max_i32 v7, v255, v3 ; encoding: [0x01,0x71,0x25,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_max_i32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_max_i32 v7, v2, v3 ; encoding: [0xff,0x71,0x25,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_max_i32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_max_i32 v7, v3, v3 ; encoding: [0x02,0x71,0x25,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_max_i32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_max_i32 v7, v4, v3 ; encoding: [0x03,0x71,0x25,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_max_i32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_max_i32 v7, s105, v3 ; encoding: [0x69,0x70,0x25,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_max_i32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_max_i32 v7, s1, v3 ; encoding: [0x01,0x70,0x25,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_max_i32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_max_i32 v7, ttmp15, v3 ; encoding: [0x7b,0x70,0x25,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_max_i32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_max_i32 v7, exec_hi, v3 ; encoding: [0x7f,0x70,0x25,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_max_i32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_max_i32 v7, exec_lo, v3 ; encoding: [0x7e,0x70,0x25,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_max_i32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_max_i32 v7, m0, v3 ; encoding: [0x7d,0x70,0x25,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_max_i32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_max_i32 v7, vcc_hi, v3 ; encoding: [0x6b,0x70,0x25,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_max_i32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_max_i32 v7, vcc_lo, v3 ; encoding: [0x6a,0x70,0x25,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_max_i32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_max_i32 v7, -1, v3 ; encoding: [0xfd,0x70,0x25,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_max_i32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_max_i32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x25,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_max_i32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_max_i32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x25,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3 ; encoding: [0x04,0x71,0x01,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3 ; encoding: [0x01,0x71,0x01,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3 ; encoding: [0xff,0x71,0x01,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3 ; encoding: [0x02,0x71,0x01,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3 ; encoding: [0x03,0x71,0x01,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3 ; encoding: [0x69,0x70,0x01,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3 ; encoding: [0x01,0x70,0x01,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x01,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x01,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x01,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3 ; encoding: [0x7d,0x70,0x01,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x01,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x01,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3 ; encoding: [0xfd,0x70,0x01,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x01,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x01,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3 ; encoding: [0x04,0x71,0x29,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3 ; encoding: [0x01,0x71,0x29,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3 ; encoding: [0xff,0x71,0x29,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3 ; encoding: [0x02,0x71,0x29,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3 ; encoding: [0x03,0x71,0x29,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3 ; encoding: [0x69,0x70,0x29,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3 ; encoding: [0x01,0x70,0x29,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x29,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x29,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x29,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3 ; encoding: [0x7d,0x70,0x29,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x29,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x29,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3 ; encoding: [0xfd,0x70,0x29,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x29,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x29,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3 ; encoding: [0x04,0x71,0x2d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3 ; encoding: [0x01,0x71,0x2d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3 ; encoding: [0xff,0x71,0x2d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3 ; encoding: [0x02,0x71,0x2d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3 ; encoding: [0x03,0x71,0x2d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3 ; encoding: [0x69,0x70,0x2d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3 ; encoding: [0x01,0x70,0x2d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x2d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x2d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x2d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3 ; encoding: [0x7d,0x70,0x2d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x2d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x2d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3 ; encoding: [0xfd,0x70,0x2d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x2d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x2d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_max_i32 v7, v1, v255
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_max_i32 v7, v1, v255 ; encoding: [0x04,0x71,0x21,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_max_i32 v7, v255, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_max_i32 v7, v255, v255 ; encoding: [0x01,0x71,0x21,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_max_i32 v7, v2, v255
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_max_i32 v7, v2, v255 ; encoding: [0xff,0x71,0x21,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_max_i32 v7, v3, v255
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_max_i32 v7, v3, v255 ; encoding: [0x02,0x71,0x21,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_max_i32 v7, v4, v255
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_max_i32 v7, v4, v255 ; encoding: [0x03,0x71,0x21,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_max_i32 v7, s1, v255
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_max_i32 v7, s1, v255 ; encoding: [0x69,0x70,0x21,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_max_i32 v7, s105, v255
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_max_i32 v7, s105, v255 ; encoding: [0x01,0x70,0x21,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_max_i32 v7, vcc_lo, v255
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_max_i32 v7, vcc_lo, v255 ; encoding: [0x7b,0x70,0x21,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_max_i32 v7, vcc_hi, v255
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_max_i32 v7, vcc_hi, v255 ; encoding: [0x7f,0x70,0x21,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_max_i32 v7, ttmp15, v255
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_max_i32 v7, ttmp15, v255 ; encoding: [0x7e,0x70,0x21,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_max_i32 v7, m0, v255
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_max_i32 v7, m0, v255 ; encoding: [0x7d,0x70,0x21,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_max_i32 v7, exec_lo, v255
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_max_i32 v7, exec_lo, v255 ; encoding: [0x6b,0x70,0x21,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_max_i32 v7, exec_hi, v255
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_max_i32 v7, exec_hi, v255 ; encoding: [0x6a,0x70,0x21,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_max_i32 v7, -1, v255
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_max_i32 v7, -1, v255 ; encoding: [0xfd,0x70,0x21,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_max_i32 v7, 0.5, v3
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_max_i32 v7, 0.5, v3 ; encoding: [0xf0,0x70,0x21,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_max_i32 v7, src_scc, v4
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_max_i32 v7, src_scc, v4 ; encoding: [0xc1,0x70,0x21,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3 ; encoding: [0x04,0x71,0x1d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3 ; encoding: [0x01,0x71,0x1d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3 ; encoding: [0xff,0x71,0x1d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3 ; encoding: [0x02,0x71,0x1d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3 ; encoding: [0x03,0x71,0x1d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3 ; encoding: [0x69,0x70,0x1d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3 ; encoding: [0x01,0x70,0x1d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x1d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x1d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x1d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3 ; encoding: [0x7d,0x70,0x1d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x1d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x1d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3 ; encoding: [0xfd,0x70,0x1d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x1d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x1d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3 ; encoding: [0x04,0x71,0x0d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3 ; encoding: [0x01,0x71,0x0d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3 ; encoding: [0xff,0x71,0x0d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3 ; encoding: [0x02,0x71,0x0d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3 ; encoding: [0x03,0x71,0x0d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3 ; encoding: [0x69,0x70,0x0d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3 ; encoding: [0x01,0x70,0x0d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x0d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x0d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x0d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3 ; encoding: [0x7d,0x70,0x0d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x0d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x0d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3 ; encoding: [0xfd,0x70,0x0d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x0d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x0d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3 ; encoding: [0x04,0x71,0x15,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3 ; encoding: [0x01,0x71,0x15,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3 ; encoding: [0xff,0x71,0x15,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3 ; encoding: [0x02,0x71,0x15,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3 ; encoding: [0x03,0x71,0x15,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3 ; encoding: [0x69,0x70,0x15,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3 ; encoding: [0x01,0x70,0x15,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x15,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x15,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x15,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3 ; encoding: [0x7d,0x70,0x15,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x15,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x15,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3 ; encoding: [0xfd,0x70,0x15,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x15,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x15,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_max_i32 v7, v1, v3 ; encoding: [0x04,0x71,0x19,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_max_i32 v7, v255, v3 ; encoding: [0x01,0x71,0x19,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_max_i32 v7, v2, v3 ; encoding: [0xff,0x71,0x19,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_max_i32 v7, v3, v3 ; encoding: [0x02,0x71,0x19,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_max_i32 v7, v4, v3 ; encoding: [0x03,0x71,0x19,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_max_i32 v7, s1, v3 ; encoding: [0x69,0x70,0x19,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_max_i32 v7, s105, v3 ; encoding: [0x01,0x70,0x19,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_max_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x70,0x19,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_max_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x70,0x19,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_max_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x70,0x19,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_max_i32 v7, m0, v3 ; encoding: [0x7d,0x70,0x19,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_max_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x70,0x19,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_max_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x70,0x19,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_max_i32 v7, -1, v3 ; encoding: [0xfd,0x70,0x19,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_max_i32 v7, 0.5, v2 ; encoding: [0xf0,0x70,0x19,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_max_i32 v7, src_scc, v5 ; encoding: [0xc1,0x70,0x19,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3 ; encoding: [0x04,0x81,0x11,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3 ; encoding: [0x01,0x81,0x11,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3 ; encoding: [0xff,0x81,0x11,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3 ; encoding: [0x02,0x81,0x11,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3 ; encoding: [0x03,0x81,0x11,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3 ; encoding: [0x69,0x80,0x11,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3 ; encoding: [0x01,0x80,0x11,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x80,0x11,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x80,0x11,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x80,0x11,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3 ; encoding: [0x7d,0x80,0x11,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x80,0x11,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x80,0x11,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3 ; encoding: [0xfd,0x80,0x11,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2 ; encoding: [0xf0,0x80,0x11,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5 ; encoding: [0xc1,0x80,0x11,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_min_i32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_min_i32 v7, v1, v3 ; encoding: [0x04,0x81,0x25,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_min_i32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_min_i32 v7, v255, v3 ; encoding: [0x01,0x81,0x25,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_min_i32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_min_i32 v7, v2, v3 ; encoding: [0xff,0x81,0x25,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_min_i32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_min_i32 v7, v3, v3 ; encoding: [0x02,0x81,0x25,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_min_i32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_min_i32 v7, v4, v3 ; encoding: [0x03,0x81,0x25,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_min_i32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_min_i32 v7, s105, v3 ; encoding: [0x69,0x80,0x25,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_min_i32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_min_i32 v7, s1, v3 ; encoding: [0x01,0x80,0x25,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_min_i32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_min_i32 v7, ttmp15, v3 ; encoding: [0x7b,0x80,0x25,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_min_i32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_min_i32 v7, exec_hi, v3 ; encoding: [0x7f,0x80,0x25,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_min_i32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_min_i32 v7, exec_lo, v3 ; encoding: [0x7e,0x80,0x25,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_min_i32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_min_i32 v7, m0, v3 ; encoding: [0x7d,0x80,0x25,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_min_i32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_min_i32 v7, vcc_hi, v3 ; encoding: [0x6b,0x80,0x25,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_min_i32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_min_i32 v7, vcc_lo, v3 ; encoding: [0x6a,0x80,0x25,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_min_i32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_min_i32 v7, -1, v3 ; encoding: [0xfd,0x80,0x25,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_min_i32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_min_i32 v7, 0.5, v2 ; encoding: [0xf0,0x80,0x25,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_min_i32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_min_i32 v7, src_scc, v5 ; encoding: [0xc1,0x80,0x25,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3 ; encoding: [0x04,0x81,0x01,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3 ; encoding: [0x01,0x81,0x01,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3 ; encoding: [0xff,0x81,0x01,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3 ; encoding: [0x02,0x81,0x01,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3 ; encoding: [0x03,0x81,0x01,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3 ; encoding: [0x69,0x80,0x01,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3 ; encoding: [0x01,0x80,0x01,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x80,0x01,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x80,0x01,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x80,0x01,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3 ; encoding: [0x7d,0x80,0x01,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x80,0x01,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x80,0x01,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3 ; encoding: [0xfd,0x80,0x01,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2 ; encoding: [0xf0,0x80,0x01,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5 ; encoding: [0xc1,0x80,0x01,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3 ; encoding: [0x04,0x81,0x29,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3 ; encoding: [0x01,0x81,0x29,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3 ; encoding: [0xff,0x81,0x29,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3 ; encoding: [0x02,0x81,0x29,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3 ; encoding: [0x03,0x81,0x29,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3 ; encoding: [0x69,0x80,0x29,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3 ; encoding: [0x01,0x80,0x29,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x80,0x29,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x80,0x29,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x80,0x29,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3 ; encoding: [0x7d,0x80,0x29,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x80,0x29,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x80,0x29,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3 ; encoding: [0xfd,0x80,0x29,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2 ; encoding: [0xf0,0x80,0x29,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5 ; encoding: [0xc1,0x80,0x29,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3 ; encoding: [0x04,0x81,0x2d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3 ; encoding: [0x01,0x81,0x2d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3 ; encoding: [0xff,0x81,0x2d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3 ; encoding: [0x02,0x81,0x2d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3 ; encoding: [0x03,0x81,0x2d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3 ; encoding: [0x69,0x80,0x2d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3 ; encoding: [0x01,0x80,0x2d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x80,0x2d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x80,0x2d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x80,0x2d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3 ; encoding: [0x7d,0x80,0x2d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x80,0x2d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x80,0x2d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3 ; encoding: [0xfd,0x80,0x2d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2 ; encoding: [0xf0,0x80,0x2d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5 ; encoding: [0xc1,0x80,0x2d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_min_i32 v7, v1, v255
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_min_i32 v7, v1, v255 ; encoding: [0x04,0x81,0x21,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_min_i32 v7, v255, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_min_i32 v7, v255, v255 ; encoding: [0x01,0x81,0x21,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_min_i32 v7, v2, v255
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_min_i32 v7, v2, v255 ; encoding: [0xff,0x81,0x21,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_min_i32 v7, v3, v255
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_min_i32 v7, v3, v255 ; encoding: [0x02,0x81,0x21,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_min_i32 v7, v4, v255
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_min_i32 v7, v4, v255 ; encoding: [0x03,0x81,0x21,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_min_i32 v7, s1, v255
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_min_i32 v7, s1, v255 ; encoding: [0x69,0x80,0x21,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_min_i32 v7, s105, v255
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_min_i32 v7, s105, v255 ; encoding: [0x01,0x80,0x21,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_min_i32 v7, vcc_lo, v255
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_min_i32 v7, vcc_lo, v255 ; encoding: [0x7b,0x80,0x21,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_min_i32 v7, vcc_hi, v255
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_min_i32 v7, vcc_hi, v255 ; encoding: [0x7f,0x80,0x21,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_min_i32 v7, ttmp15, v255
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_min_i32 v7, ttmp15, v255 ; encoding: [0x7e,0x80,0x21,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_min_i32 v7, m0, v255
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_min_i32 v7, m0, v255 ; encoding: [0x7d,0x80,0x21,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_min_i32 v7, exec_lo, v255
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_min_i32 v7, exec_lo, v255 ; encoding: [0x6b,0x80,0x21,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_min_i32 v7, exec_hi, v255
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_min_i32 v7, exec_hi, v255 ; encoding: [0x6a,0x80,0x21,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_min_i32 v7, -1, v255
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_min_i32 v7, -1, v255 ; encoding: [0xfd,0x80,0x21,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_min_i32 v7, 0.5, v3
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_min_i32 v7, 0.5, v3 ; encoding: [0xf0,0x80,0x21,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_min_i32 v7, src_scc, v4
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_min_i32 v7, src_scc, v4 ; encoding: [0xc1,0x80,0x21,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3 ; encoding: [0x04,0x81,0x1d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3 ; encoding: [0x01,0x81,0x1d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3 ; encoding: [0xff,0x81,0x1d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3 ; encoding: [0x02,0x81,0x1d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3 ; encoding: [0x03,0x81,0x1d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3 ; encoding: [0x69,0x80,0x1d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3 ; encoding: [0x01,0x80,0x1d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x80,0x1d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x80,0x1d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x80,0x1d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3 ; encoding: [0x7d,0x80,0x1d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x80,0x1d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x80,0x1d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3 ; encoding: [0xfd,0x80,0x1d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2 ; encoding: [0xf0,0x80,0x1d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5 ; encoding: [0xc1,0x80,0x1d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3 ; encoding: [0x04,0x81,0x0d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3 ; encoding: [0x01,0x81,0x0d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3 ; encoding: [0xff,0x81,0x0d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3 ; encoding: [0x02,0x81,0x0d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3 ; encoding: [0x03,0x81,0x0d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3 ; encoding: [0x69,0x80,0x0d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3 ; encoding: [0x01,0x80,0x0d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x80,0x0d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x80,0x0d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x80,0x0d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3 ; encoding: [0x7d,0x80,0x0d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x80,0x0d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x80,0x0d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3 ; encoding: [0xfd,0x80,0x0d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2 ; encoding: [0xf0,0x80,0x0d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5 ; encoding: [0xc1,0x80,0x0d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3 ; encoding: [0x04,0x81,0x15,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3 ; encoding: [0x01,0x81,0x15,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3 ; encoding: [0xff,0x81,0x15,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3 ; encoding: [0x02,0x81,0x15,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3 ; encoding: [0x03,0x81,0x15,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3 ; encoding: [0x69,0x80,0x15,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3 ; encoding: [0x01,0x80,0x15,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x80,0x15,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x80,0x15,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x80,0x15,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3 ; encoding: [0x7d,0x80,0x15,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x80,0x15,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x80,0x15,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3 ; encoding: [0xfd,0x80,0x15,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2 ; encoding: [0xf0,0x80,0x15,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5 ; encoding: [0xc1,0x80,0x15,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_min_i32 v7, v1, v3 ; encoding: [0x04,0x81,0x19,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_min_i32 v7, v255, v3 ; encoding: [0x01,0x81,0x19,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_min_i32 v7, v2, v3 ; encoding: [0xff,0x81,0x19,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_min_i32 v7, v3, v3 ; encoding: [0x02,0x81,0x19,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_min_i32 v7, v4, v3 ; encoding: [0x03,0x81,0x19,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_min_i32 v7, s1, v3 ; encoding: [0x69,0x80,0x19,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_min_i32 v7, s105, v3 ; encoding: [0x01,0x80,0x19,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_min_i32 v7, vcc_lo, v3 ; encoding: [0x7b,0x80,0x19,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_min_i32 v7, vcc_hi, v3 ; encoding: [0x7f,0x80,0x19,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_min_i32 v7, ttmp15, v3 ; encoding: [0x7e,0x80,0x19,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_min_i32 v7, m0, v3 ; encoding: [0x7d,0x80,0x19,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_min_i32 v7, exec_lo, v3 ; encoding: [0x6b,0x80,0x19,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_min_i32 v7, exec_hi, v3 ; encoding: [0x6a,0x80,0x19,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_min_i32 v7, -1, v3 ; encoding: [0xfd,0x80,0x19,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_min_i32 v7, 0.5, v2 ; encoding: [0xf0,0x80,0x19,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_min_i32 v7, src_scc, v5 ; encoding: [0xc1,0x80,0x19,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3 ; encoding: [0x04,0x41,0x11,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3 ; encoding: [0x01,0x41,0x11,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3 ; encoding: [0xff,0x41,0x11,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3 ; encoding: [0x02,0x41,0x11,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3 ; encoding: [0x03,0x41,0x11,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3 ; encoding: [0x69,0x40,0x11,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3 ; encoding: [0x01,0x40,0x11,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x11,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x11,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x11,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x40,0x11,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x11,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x11,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x40,0x11,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x11,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x11,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_sub_nc_u32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_sub_nc_u32 v7, v1, v3 ; encoding: [0x04,0x41,0x25,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_sub_nc_u32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_sub_nc_u32 v7, v255, v3 ; encoding: [0x01,0x41,0x25,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_sub_nc_u32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_sub_nc_u32 v7, v2, v3 ; encoding: [0xff,0x41,0x25,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_sub_nc_u32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_sub_nc_u32 v7, v3, v3 ; encoding: [0x02,0x41,0x25,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_sub_nc_u32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_sub_nc_u32 v7, v4, v3 ; encoding: [0x03,0x41,0x25,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_sub_nc_u32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_sub_nc_u32 v7, s105, v3 ; encoding: [0x69,0x40,0x25,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_sub_nc_u32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_sub_nc_u32 v7, s1, v3 ; encoding: [0x01,0x40,0x25,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_sub_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_sub_nc_u32 v7, ttmp15, v3 ; encoding: [0x7b,0x40,0x25,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_sub_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_sub_nc_u32 v7, exec_hi, v3 ; encoding: [0x7f,0x40,0x25,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_sub_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_sub_nc_u32 v7, exec_lo, v3 ; encoding: [0x7e,0x40,0x25,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_sub_nc_u32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_sub_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x40,0x25,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_sub_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_sub_nc_u32 v7, vcc_hi, v3 ; encoding: [0x6b,0x40,0x25,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_sub_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_sub_nc_u32 v7, vcc_lo, v3 ; encoding: [0x6a,0x40,0x25,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_sub_nc_u32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_sub_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x40,0x25,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_sub_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_sub_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x25,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_sub_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_sub_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x25,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3 ; encoding: [0x04,0x41,0x01,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3 ; encoding: [0x01,0x41,0x01,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3 ; encoding: [0xff,0x41,0x01,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3 ; encoding: [0x02,0x41,0x01,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3 ; encoding: [0x03,0x41,0x01,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3 ; encoding: [0x69,0x40,0x01,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3 ; encoding: [0x01,0x40,0x01,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x01,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x01,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x01,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x40,0x01,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x01,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x01,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x40,0x01,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x01,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x01,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3 ; encoding: [0x04,0x41,0x29,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3 ; encoding: [0x01,0x41,0x29,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3 ; encoding: [0xff,0x41,0x29,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3 ; encoding: [0x02,0x41,0x29,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3 ; encoding: [0x03,0x41,0x29,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3 ; encoding: [0x69,0x40,0x29,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3 ; encoding: [0x01,0x40,0x29,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x29,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x29,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x29,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x40,0x29,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x29,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x29,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x40,0x29,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x29,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x29,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3
// GFX1250: v_dual_min_num_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3 ; encoding: [0x04,0x41,0x2d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3
// GFX1250: v_dual_min_num_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3 ; encoding: [0x01,0x41,0x2d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3
// GFX1250: v_dual_min_num_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3 ; encoding: [0xff,0x41,0x2d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3
// GFX1250: v_dual_min_num_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3 ; encoding: [0x02,0x41,0x2d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3
// GFX1250: v_dual_min_num_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3 ; encoding: [0x03,0x41,0x2d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3
// GFX1250: v_dual_min_num_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3 ; encoding: [0x69,0x40,0x2d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3
// GFX1250: v_dual_min_num_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3 ; encoding: [0x01,0x40,0x2d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_min_num_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x2d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_min_num_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x2d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_min_num_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x2d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3
// GFX1250: v_dual_min_num_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x40,0x2d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x2d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_min_num_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x2d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3
// GFX1250: v_dual_min_num_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x40,0x2d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_min_num_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x2d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_min_num_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_min_num_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x2d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v4 :: v_dual_sub_nc_u32 v7, v1, v255
// GFX1250: v_dual_mov_b32 v255, v4 :: v_dual_sub_nc_u32 v7, v1, v255 ; encoding: [0x04,0x41,0x21,0xcf,0x01,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v1 :: v_dual_sub_nc_u32 v7, v255, v255
// GFX1250: v_dual_mov_b32 v255, v1 :: v_dual_sub_nc_u32 v7, v255, v255 ; encoding: [0x01,0x41,0x21,0xcf,0xff,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v255 :: v_dual_sub_nc_u32 v7, v2, v255
// GFX1250: v_dual_mov_b32 v255, v255 :: v_dual_sub_nc_u32 v7, v2, v255 ; encoding: [0xff,0x41,0x21,0xcf,0x02,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v2 :: v_dual_sub_nc_u32 v7, v3, v255
// GFX1250: v_dual_mov_b32 v255, v2 :: v_dual_sub_nc_u32 v7, v3, v255 ; encoding: [0x02,0x41,0x21,0xcf,0x03,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, v3 :: v_dual_sub_nc_u32 v7, v4, v255
// GFX1250: v_dual_mov_b32 v255, v3 :: v_dual_sub_nc_u32 v7, v4, v255 ; encoding: [0x03,0x41,0x21,0xcf,0x04,0x01,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s105 :: v_dual_sub_nc_u32 v7, s1, v255
// GFX1250: v_dual_mov_b32 v255, s105 :: v_dual_sub_nc_u32 v7, s1, v255 ; encoding: [0x69,0x40,0x21,0xcf,0x01,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, s1 :: v_dual_sub_nc_u32 v7, s105, v255
// GFX1250: v_dual_mov_b32 v255, s1 :: v_dual_sub_nc_u32 v7, s105, v255 ; encoding: [0x01,0x40,0x21,0xcf,0x69,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, ttmp15 :: v_dual_sub_nc_u32 v7, vcc_lo, v255
// GFX1250: v_dual_mov_b32 v255, ttmp15 :: v_dual_sub_nc_u32 v7, vcc_lo, v255 ; encoding: [0x7b,0x40,0x21,0xcf,0x6a,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_hi :: v_dual_sub_nc_u32 v7, vcc_hi, v255
// GFX1250: v_dual_mov_b32 v255, exec_hi :: v_dual_sub_nc_u32 v7, vcc_hi, v255 ; encoding: [0x7f,0x40,0x21,0xcf,0x6b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, exec_lo :: v_dual_sub_nc_u32 v7, ttmp15, v255
// GFX1250: v_dual_mov_b32 v255, exec_lo :: v_dual_sub_nc_u32 v7, ttmp15, v255 ; encoding: [0x7e,0x40,0x21,0xcf,0x7b,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, m0 :: v_dual_sub_nc_u32 v7, m0, v255
// GFX1250: v_dual_mov_b32 v255, m0 :: v_dual_sub_nc_u32 v7, m0, v255 ; encoding: [0x7d,0x40,0x21,0xcf,0x7d,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_hi :: v_dual_sub_nc_u32 v7, exec_lo, v255
// GFX1250: v_dual_mov_b32 v255, vcc_hi :: v_dual_sub_nc_u32 v7, exec_lo, v255 ; encoding: [0x6b,0x40,0x21,0xcf,0x7e,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, vcc_lo :: v_dual_sub_nc_u32 v7, exec_hi, v255
// GFX1250: v_dual_mov_b32 v255, vcc_lo :: v_dual_sub_nc_u32 v7, exec_hi, v255 ; encoding: [0x6a,0x40,0x21,0xcf,0x7f,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, src_scc :: v_dual_sub_nc_u32 v7, -1, v255
// GFX1250: v_dual_mov_b32 v255, src_scc :: v_dual_sub_nc_u32 v7, -1, v255 ; encoding: [0xfd,0x40,0x21,0xcf,0xc1,0x00,0x00,0x00,0xff,0xff,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, 0.5 :: v_dual_sub_nc_u32 v7, 0.5, v3
// GFX1250: v_dual_mov_b32 v255, 0.5 :: v_dual_sub_nc_u32 v7, 0.5, v3 ; encoding: [0xf0,0x40,0x21,0xcf,0xf0,0x00,0x00,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mov_b32 v255, -1 :: v_dual_sub_nc_u32 v7, src_scc, v4
// GFX1250: v_dual_mov_b32 v255, -1 :: v_dual_sub_nc_u32 v7, src_scc, v4 ; encoding: [0xc1,0x40,0x21,0xcf,0xfd,0x00,0x00,0x00,0xff,0x04,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3 ; encoding: [0x04,0x41,0x1d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3 ; encoding: [0x01,0x41,0x1d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3 ; encoding: [0xff,0x41,0x1d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3 ; encoding: [0x02,0x41,0x1d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3 ; encoding: [0x03,0x41,0x1d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3 ; encoding: [0x69,0x40,0x1d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3 ; encoding: [0x01,0x40,0x1d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x1d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x1d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x1d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x40,0x1d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x1d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x1d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3
// GFX1250: v_dual_mul_dx9_zero_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x40,0x1d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_mul_dx9_zero_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x1d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_mul_dx9_zero_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x1d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3
// GFX1250: v_dual_mul_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3 ; encoding: [0x04,0x41,0x0d,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3
// GFX1250: v_dual_mul_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3 ; encoding: [0x01,0x41,0x0d,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3
// GFX1250: v_dual_mul_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3 ; encoding: [0xff,0x41,0x0d,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3
// GFX1250: v_dual_mul_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3 ; encoding: [0x02,0x41,0x0d,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3
// GFX1250: v_dual_mul_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3 ; encoding: [0x03,0x41,0x0d,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3
// GFX1250: v_dual_mul_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3 ; encoding: [0x69,0x40,0x0d,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3
// GFX1250: v_dual_mul_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3 ; encoding: [0x01,0x40,0x0d,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_mul_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x0d,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_mul_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x0d,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_mul_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x0d,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3
// GFX1250: v_dual_mul_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x40,0x0d,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_mul_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x0d,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_mul_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x0d,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3
// GFX1250: v_dual_mul_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x40,0x0d,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_mul_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x0d,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_mul_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_mul_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x0d,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3
// GFX1250: v_dual_sub_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3 ; encoding: [0x04,0x41,0x15,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3
// GFX1250: v_dual_sub_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3 ; encoding: [0x01,0x41,0x15,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3
// GFX1250: v_dual_sub_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3 ; encoding: [0xff,0x41,0x15,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3
// GFX1250: v_dual_sub_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3 ; encoding: [0x02,0x41,0x15,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3
// GFX1250: v_dual_sub_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3 ; encoding: [0x03,0x41,0x15,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3
// GFX1250: v_dual_sub_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3 ; encoding: [0x69,0x40,0x15,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3
// GFX1250: v_dual_sub_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3 ; encoding: [0x01,0x40,0x15,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_sub_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x15,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_sub_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x15,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_sub_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x15,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3
// GFX1250: v_dual_sub_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x40,0x15,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_sub_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x15,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_sub_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x15,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3
// GFX1250: v_dual_sub_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x40,0x15,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_sub_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x15,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_sub_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_sub_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x15,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3
// GFX1250: v_dual_subrev_f32 v255, v4, v2 :: v_dual_sub_nc_u32 v7, v1, v3 ; encoding: [0x04,0x41,0x19,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3
// GFX1250: v_dual_subrev_f32 v255, v1, v2 :: v_dual_sub_nc_u32 v7, v255, v3 ; encoding: [0x01,0x41,0x19,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3
// GFX1250: v_dual_subrev_f32 v255, v255, v2 :: v_dual_sub_nc_u32 v7, v2, v3 ; encoding: [0xff,0x41,0x19,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3
// GFX1250: v_dual_subrev_f32 v255, v2, v2 :: v_dual_sub_nc_u32 v7, v3, v3 ; encoding: [0x02,0x41,0x19,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3
// GFX1250: v_dual_subrev_f32 v255, v3, v2 :: v_dual_sub_nc_u32 v7, v4, v3 ; encoding: [0x03,0x41,0x19,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3
// GFX1250: v_dual_subrev_f32 v255, s105, v2 :: v_dual_sub_nc_u32 v7, s1, v3 ; encoding: [0x69,0x40,0x19,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3
// GFX1250: v_dual_subrev_f32 v255, s1, v2 :: v_dual_sub_nc_u32 v7, s105, v3 ; encoding: [0x01,0x40,0x19,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3
// GFX1250: v_dual_subrev_f32 v255, ttmp15, v2 :: v_dual_sub_nc_u32 v7, vcc_lo, v3 ; encoding: [0x7b,0x40,0x19,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3
// GFX1250: v_dual_subrev_f32 v255, exec_hi, v2 :: v_dual_sub_nc_u32 v7, vcc_hi, v3 ; encoding: [0x7f,0x40,0x19,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3
// GFX1250: v_dual_subrev_f32 v255, exec_lo, v2 :: v_dual_sub_nc_u32 v7, ttmp15, v3 ; encoding: [0x7e,0x40,0x19,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3
// GFX1250: v_dual_subrev_f32 v255, m0, v2 :: v_dual_sub_nc_u32 v7, m0, v3 ; encoding: [0x7d,0x40,0x19,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_hi, v2 :: v_dual_sub_nc_u32 v7, exec_lo, v3 ; encoding: [0x6b,0x40,0x19,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3
// GFX1250: v_dual_subrev_f32 v255, vcc_lo, v2 :: v_dual_sub_nc_u32 v7, exec_hi, v3 ; encoding: [0x6a,0x40,0x19,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3
// GFX1250: v_dual_subrev_f32 v255, src_scc, v2 :: v_dual_sub_nc_u32 v7, -1, v3 ; encoding: [0xfd,0x40,0x19,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2
// GFX1250: v_dual_subrev_f32 v255, 0.5, v3 :: v_dual_sub_nc_u32 v7, 0.5, v2 ; encoding: [0xf0,0x40,0x19,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_subrev_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5
// GFX1250: v_dual_subrev_f32 v255, -1, v4 :: v_dual_sub_nc_u32 v7, src_scc, v5 ; encoding: [0xc1,0x40,0x19,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v4, v2 :: v_dual_lshrrev_b32 v7, v1, v3
// GFX1250: v_dual_add_f32 v255, v4, v2 :: v_dual_lshrrev_b32 v7, v1, v3 ; encoding: [0x04,0x51,0x11,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v1, v2 :: v_dual_lshrrev_b32 v7, v255, v3
// GFX1250: v_dual_add_f32 v255, v1, v2 :: v_dual_lshrrev_b32 v7, v255, v3 ; encoding: [0x01,0x51,0x11,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v255, v2 :: v_dual_lshrrev_b32 v7, v2, v3
// GFX1250: v_dual_add_f32 v255, v255, v2 :: v_dual_lshrrev_b32 v7, v2, v3 ; encoding: [0xff,0x51,0x11,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v2, v2 :: v_dual_lshrrev_b32 v7, v3, v3
// GFX1250: v_dual_add_f32 v255, v2, v2 :: v_dual_lshrrev_b32 v7, v3, v3 ; encoding: [0x02,0x51,0x11,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, v3, v2 :: v_dual_lshrrev_b32 v7, v4, v3
// GFX1250: v_dual_add_f32 v255, v3, v2 :: v_dual_lshrrev_b32 v7, v4, v3 ; encoding: [0x03,0x51,0x11,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s105, v2 :: v_dual_lshrrev_b32 v7, s1, v3
// GFX1250: v_dual_add_f32 v255, s105, v2 :: v_dual_lshrrev_b32 v7, s1, v3 ; encoding: [0x69,0x50,0x11,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, s1, v2 :: v_dual_lshrrev_b32 v7, s105, v3
// GFX1250: v_dual_add_f32 v255, s1, v2 :: v_dual_lshrrev_b32 v7, s105, v3 ; encoding: [0x01,0x50,0x11,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, ttmp15, v2 :: v_dual_lshrrev_b32 v7, vcc_lo, v3
// GFX1250: v_dual_add_f32 v255, ttmp15, v2 :: v_dual_lshrrev_b32 v7, vcc_lo, v3 ; encoding: [0x7b,0x50,0x11,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_hi, v2 :: v_dual_lshrrev_b32 v7, vcc_hi, v3
// GFX1250: v_dual_add_f32 v255, exec_hi, v2 :: v_dual_lshrrev_b32 v7, vcc_hi, v3 ; encoding: [0x7f,0x50,0x11,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, exec_lo, v2 :: v_dual_lshrrev_b32 v7, ttmp15, v3
// GFX1250: v_dual_add_f32 v255, exec_lo, v2 :: v_dual_lshrrev_b32 v7, ttmp15, v3 ; encoding: [0x7e,0x50,0x11,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, m0, v2 :: v_dual_lshrrev_b32 v7, m0, v3
// GFX1250: v_dual_add_f32 v255, m0, v2 :: v_dual_lshrrev_b32 v7, m0, v3 ; encoding: [0x7d,0x50,0x11,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_lshrrev_b32 v7, exec_lo, v3
// GFX1250: v_dual_add_f32 v255, vcc_hi, v2 :: v_dual_lshrrev_b32 v7, exec_lo, v3 ; encoding: [0x6b,0x50,0x11,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_lshrrev_b32 v7, exec_hi, v3
// GFX1250: v_dual_add_f32 v255, vcc_lo, v2 :: v_dual_lshrrev_b32 v7, exec_hi, v3 ; encoding: [0x6a,0x50,0x11,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, src_scc, v2 :: v_dual_lshrrev_b32 v7, -1, v3
// GFX1250: v_dual_add_f32 v255, src_scc, v2 :: v_dual_lshrrev_b32 v7, -1, v3 ; encoding: [0xfd,0x50,0x11,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, 0.5, v3 :: v_dual_lshrrev_b32 v7, 0.5, v2
// GFX1250: v_dual_add_f32 v255, 0.5, v3 :: v_dual_lshrrev_b32 v7, 0.5, v2 ; encoding: [0xf0,0x50,0x11,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_add_f32 v255, -1, v4 :: v_dual_lshrrev_b32 v7, src_scc, v5
// GFX1250: v_dual_add_f32 v255, -1, v4 :: v_dual_lshrrev_b32 v7, src_scc, v5 ; encoding: [0xc1,0x50,0x11,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_lshrrev_b32 v7, v1, v3
// GFX1250: v_dual_cndmask_b32 v255, v4, v2, vcc_lo :: v_dual_lshrrev_b32 v7, v1, v3 ; encoding: [0x04,0x51,0x25,0xcf,0x01,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_lshrrev_b32 v7, v255, v3
// GFX1250: v_dual_cndmask_b32 v255, v1, v2, vcc_lo :: v_dual_lshrrev_b32 v7, v255, v3 ; encoding: [0x01,0x51,0x25,0xcf,0xff,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_lshrrev_b32 v7, v2, v3
// GFX1250: v_dual_cndmask_b32 v255, v255, v2, vcc_lo :: v_dual_lshrrev_b32 v7, v2, v3 ; encoding: [0xff,0x51,0x25,0xcf,0x02,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_lshrrev_b32 v7, v3, v3
// GFX1250: v_dual_cndmask_b32 v255, v2, v2, vcc_lo :: v_dual_lshrrev_b32 v7, v3, v3 ; encoding: [0x02,0x51,0x25,0xcf,0x03,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_lshrrev_b32 v7, v4, v3
// GFX1250: v_dual_cndmask_b32 v255, v3, v2, vcc_lo :: v_dual_lshrrev_b32 v7, v4, v3 ; encoding: [0x03,0x51,0x25,0xcf,0x04,0x01,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_lshrrev_b32 v7, s105, v3
// GFX1250: v_dual_cndmask_b32 v255, s105, v2, vcc_lo :: v_dual_lshrrev_b32 v7, s105, v3 ; encoding: [0x69,0x50,0x25,0xcf,0x69,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_lshrrev_b32 v7, s1, v3
// GFX1250: v_dual_cndmask_b32 v255, s1, v2, vcc_lo :: v_dual_lshrrev_b32 v7, s1, v3 ; encoding: [0x01,0x50,0x25,0xcf,0x01,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_lshrrev_b32 v7, ttmp15, v3
// GFX1250: v_dual_cndmask_b32 v255, ttmp15, v2, vcc_lo :: v_dual_lshrrev_b32 v7, ttmp15, v3 ; encoding: [0x7b,0x50,0x25,0xcf,0x7b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_lshrrev_b32 v7, exec_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_hi, v2, vcc_lo :: v_dual_lshrrev_b32 v7, exec_hi, v3 ; encoding: [0x7f,0x50,0x25,0xcf,0x7f,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_lshrrev_b32 v7, exec_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, exec_lo, v2, vcc_lo :: v_dual_lshrrev_b32 v7, exec_lo, v3 ; encoding: [0x7e,0x50,0x25,0xcf,0x7e,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_lshrrev_b32 v7, m0, v3
// GFX1250: v_dual_cndmask_b32 v255, m0, v2, vcc_lo :: v_dual_lshrrev_b32 v7, m0, v3 ; encoding: [0x7d,0x50,0x25,0xcf,0x7d,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_lshrrev_b32 v7, vcc_hi, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_hi, v2, vcc_lo :: v_dual_lshrrev_b32 v7, vcc_hi, v3 ; encoding: [0x6b,0x50,0x25,0xcf,0x6b,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_lshrrev_b32 v7, vcc_lo, v3
// GFX1250: v_dual_cndmask_b32 v255, vcc_lo, v2, vcc_lo :: v_dual_lshrrev_b32 v7, vcc_lo, v3 ; encoding: [0x6a,0x50,0x25,0xcf,0x6a,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_lshrrev_b32 v7, -1, v3
// GFX1250: v_dual_cndmask_b32 v255, src_scc, v2, vcc_lo :: v_dual_lshrrev_b32 v7, -1, v3 ; encoding: [0xfd,0x50,0x25,0xcf,0xc1,0x00,0x02,0x6a,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_lshrrev_b32 v7, 0.5, v2
// GFX1250: v_dual_cndmask_b32 v255, 0.5, v3, vcc_lo :: v_dual_lshrrev_b32 v7, 0.5, v2 ; encoding: [0xf0,0x50,0x25,0xcf,0xf0,0x00,0x03,0x6a,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_lshrrev_b32 v7, src_scc, v5
// GFX1250: v_dual_cndmask_b32 v255, -1, v4, vcc_lo :: v_dual_lshrrev_b32 v7, src_scc, v5 ; encoding: [0xc1,0x50,0x25,0xcf,0xfd,0x00,0x04,0x6a,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v4, v2 :: v_dual_lshrrev_b32 v7, v1, v3
// GFX1250: v_dual_fmac_f32 v255, v4, v2 :: v_dual_lshrrev_b32 v7, v1, v3 ; encoding: [0x04,0x51,0x01,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v1, v2 :: v_dual_lshrrev_b32 v7, v255, v3
// GFX1250: v_dual_fmac_f32 v255, v1, v2 :: v_dual_lshrrev_b32 v7, v255, v3 ; encoding: [0x01,0x51,0x01,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v255, v2 :: v_dual_lshrrev_b32 v7, v2, v3
// GFX1250: v_dual_fmac_f32 v255, v255, v2 :: v_dual_lshrrev_b32 v7, v2, v3 ; encoding: [0xff,0x51,0x01,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v2, v2 :: v_dual_lshrrev_b32 v7, v3, v3
// GFX1250: v_dual_fmac_f32 v255, v2, v2 :: v_dual_lshrrev_b32 v7, v3, v3 ; encoding: [0x02,0x51,0x01,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, v3, v2 :: v_dual_lshrrev_b32 v7, v4, v3
// GFX1250: v_dual_fmac_f32 v255, v3, v2 :: v_dual_lshrrev_b32 v7, v4, v3 ; encoding: [0x03,0x51,0x01,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s105, v2 :: v_dual_lshrrev_b32 v7, s1, v3
// GFX1250: v_dual_fmac_f32 v255, s105, v2 :: v_dual_lshrrev_b32 v7, s1, v3 ; encoding: [0x69,0x50,0x01,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, s1, v2 :: v_dual_lshrrev_b32 v7, s105, v3
// GFX1250: v_dual_fmac_f32 v255, s1, v2 :: v_dual_lshrrev_b32 v7, s105, v3 ; encoding: [0x01,0x50,0x01,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_lshrrev_b32 v7, vcc_lo, v3
// GFX1250: v_dual_fmac_f32 v255, ttmp15, v2 :: v_dual_lshrrev_b32 v7, vcc_lo, v3 ; encoding: [0x7b,0x50,0x01,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_lshrrev_b32 v7, vcc_hi, v3
// GFX1250: v_dual_fmac_f32 v255, exec_hi, v2 :: v_dual_lshrrev_b32 v7, vcc_hi, v3 ; encoding: [0x7f,0x50,0x01,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_lshrrev_b32 v7, ttmp15, v3
// GFX1250: v_dual_fmac_f32 v255, exec_lo, v2 :: v_dual_lshrrev_b32 v7, ttmp15, v3 ; encoding: [0x7e,0x50,0x01,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, m0, v2 :: v_dual_lshrrev_b32 v7, m0, v3
// GFX1250: v_dual_fmac_f32 v255, m0, v2 :: v_dual_lshrrev_b32 v7, m0, v3 ; encoding: [0x7d,0x50,0x01,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_lshrrev_b32 v7, exec_lo, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_hi, v2 :: v_dual_lshrrev_b32 v7, exec_lo, v3 ; encoding: [0x6b,0x50,0x01,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_lshrrev_b32 v7, exec_hi, v3
// GFX1250: v_dual_fmac_f32 v255, vcc_lo, v2 :: v_dual_lshrrev_b32 v7, exec_hi, v3 ; encoding: [0x6a,0x50,0x01,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_lshrrev_b32 v7, -1, v3
// GFX1250: v_dual_fmac_f32 v255, src_scc, v2 :: v_dual_lshrrev_b32 v7, -1, v3 ; encoding: [0xfd,0x50,0x01,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_lshrrev_b32 v7, 0.5, v2
// GFX1250: v_dual_fmac_f32 v255, 0.5, v3 :: v_dual_lshrrev_b32 v7, 0.5, v2 ; encoding: [0xf0,0x50,0x01,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_fmac_f32 v255, -1, v4 :: v_dual_lshrrev_b32 v7, src_scc, v5
// GFX1250: v_dual_fmac_f32 v255, -1, v4 :: v_dual_lshrrev_b32 v7, src_scc, v5 ; encoding: [0xc1,0x50,0x01,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v4, v2 :: v_dual_lshrrev_b32 v7, v1, v3
// GFX1250: v_dual_max_num_f32 v255, v4, v2 :: v_dual_lshrrev_b32 v7, v1, v3 ; encoding: [0x04,0x51,0x29,0xcf,0x01,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v1, v2 :: v_dual_lshrrev_b32 v7, v255, v3
// GFX1250: v_dual_max_num_f32 v255, v1, v2 :: v_dual_lshrrev_b32 v7, v255, v3 ; encoding: [0x01,0x51,0x29,0xcf,0xff,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v255, v2 :: v_dual_lshrrev_b32 v7, v2, v3
// GFX1250: v_dual_max_num_f32 v255, v255, v2 :: v_dual_lshrrev_b32 v7, v2, v3 ; encoding: [0xff,0x51,0x29,0xcf,0x02,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v2, v2 :: v_dual_lshrrev_b32 v7, v3, v3
// GFX1250: v_dual_max_num_f32 v255, v2, v2 :: v_dual_lshrrev_b32 v7, v3, v3 ; encoding: [0x02,0x51,0x29,0xcf,0x03,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, v3, v2 :: v_dual_lshrrev_b32 v7, v4, v3
// GFX1250: v_dual_max_num_f32 v255, v3, v2 :: v_dual_lshrrev_b32 v7, v4, v3 ; encoding: [0x03,0x51,0x29,0xcf,0x04,0x01,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s105, v2 :: v_dual_lshrrev_b32 v7, s1, v3
// GFX1250: v_dual_max_num_f32 v255, s105, v2 :: v_dual_lshrrev_b32 v7, s1, v3 ; encoding: [0x69,0x50,0x29,0xcf,0x01,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, s1, v2 :: v_dual_lshrrev_b32 v7, s105, v3
// GFX1250: v_dual_max_num_f32 v255, s1, v2 :: v_dual_lshrrev_b32 v7, s105, v3 ; encoding: [0x01,0x50,0x29,0xcf,0x69,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_lshrrev_b32 v7, vcc_lo, v3
// GFX1250: v_dual_max_num_f32 v255, ttmp15, v2 :: v_dual_lshrrev_b32 v7, vcc_lo, v3 ; encoding: [0x7b,0x50,0x29,0xcf,0x6a,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_lshrrev_b32 v7, vcc_hi, v3
// GFX1250: v_dual_max_num_f32 v255, exec_hi, v2 :: v_dual_lshrrev_b32 v7, vcc_hi, v3 ; encoding: [0x7f,0x50,0x29,0xcf,0x6b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_lshrrev_b32 v7, ttmp15, v3
// GFX1250: v_dual_max_num_f32 v255, exec_lo, v2 :: v_dual_lshrrev_b32 v7, ttmp15, v3 ; encoding: [0x7e,0x50,0x29,0xcf,0x7b,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, m0, v2 :: v_dual_lshrrev_b32 v7, m0, v3
// GFX1250: v_dual_max_num_f32 v255, m0, v2 :: v_dual_lshrrev_b32 v7, m0, v3 ; encoding: [0x7d,0x50,0x29,0xcf,0x7d,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_lshrrev_b32 v7, exec_lo, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_hi, v2 :: v_dual_lshrrev_b32 v7, exec_lo, v3 ; encoding: [0x6b,0x50,0x29,0xcf,0x7e,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_lshrrev_b32 v7, exec_hi, v3
// GFX1250: v_dual_max_num_f32 v255, vcc_lo, v2 :: v_dual_lshrrev_b32 v7, exec_hi, v3 ; encoding: [0x6a,0x50,0x29,0xcf,0x7f,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_lshrrev_b32 v7, -1, v3
// GFX1250: v_dual_max_num_f32 v255, src_scc, v2 :: v_dual_lshrrev_b32 v7, -1, v3 ; encoding: [0xfd,0x50,0x29,0xcf,0xc1,0x00,0x02,0x00,0xff,0x03,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_lshrrev_b32 v7, 0.5, v2
// GFX1250: v_dual_max_num_f32 v255, 0.5, v3 :: v_dual_lshrrev_b32 v7, 0.5, v2 ; encoding: [0xf0,0x50,0x29,0xcf,0xf0,0x00,0x03,0x00,0xff,0x02,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_dual_max_num_f32 v255, -1, v4 :: v_dual_lshrrev_b32 v7, src_scc, v5
// GFX1250: v_dual_max_num_f32 v255, -1, v4 :: v_dual_lshrrev_b32 v7, src_scc, v5 ; encoding: [0xc1,0x50,0x29,0xcf,0xfd,0x00,0x04,0x00,0xff,0x05,0x00,0x07]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error
