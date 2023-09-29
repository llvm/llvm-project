// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck -check-prefixes=GCN,GFX10 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck -check-prefixes=GCN-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck -check-prefixes=GCN,GFX11 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck -check-prefixes=GCN-ERR,GFX11-ERR --implicit-check-not=error: %s

//-----------------------------------------------------------------------------------------
// On GFX10 we can use two scalar operands (except for 64-bit shift instructions)

v_add_f32 v0, s0, s1
// GCN: v_add_f32_e64 v0, s0, s1        ; encoding: [0x00,0x00,0x03,0xd5,0x00,0x02,0x00,0x00]

v_madak_f32 v0, s0, v1, 42.42
// GFX10: v_madak_f32 v0, s0, v1, 0x4229ae14 ; encoding: [0x00,0x02,0x00,0x42,0x14,0xae,0x29,0x42]
// GFX11-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_med3_f32 v0, s0, s0, s1
// GFX10: v_med3_f32 v0, s0, s0, s1       ; encoding: [0x00,0x00,0x57,0xd5,0x00,0x00,0x04,0x00]
// GFX11: v_med3_f32 v0, s0, s0, s1       ; encoding: [0x00,0x00,0x1f,0xd6,0x00,0x00,0x04,0x00]

//-----------------------------------------------------------------------------------------
// 64-bit shift instructions can use only one scalar value input

v_ashrrev_i64 v[0:1], 0x100, s[0:1]
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_ashrrev_i64 v[0:1], s2, s[0:1]
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

//-----------------------------------------------------------------------------------------
// v_div_fmas implicitly reads VCC, so only one scalar operand is possible

v_div_fmas_f32 v5, s3, s3, s3
// GFX10: v_div_fmas_f32 v5, s3, s3, s3   ; encoding: [0x05,0x00,0x6f,0xd5,0x03,0x06,0x0c,0x00]
// GFX11: v_div_fmas_f32 v5, s3, s3, s3   ; encoding: [0x05,0x00,0x37,0xd6,0x03,0x06,0x0c,0x00]

v_div_fmas_f32 v5, s3, s3, s2
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f32 v5, s3, 0x123, v3
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f64 v[5:6], 0x12345678, 0x12345678, 0x12345678
// GFX10: v_div_fmas_f64 v[5:6], 0x12345678, 0x12345678, 0x12345678 ; encoding: [0x05,0x00,0x70,0xd5,0xff,0xfe,0xfd,0x03,0x78,0x56,0x34,0x12]
// GFX11: v_div_fmas_f64 v[5:6], 0x12345678, 0x12345678, 0x12345678 ; encoding: [0x05,0x00,0x38,0xd6,0xff,0xfe,0xfd,0x03,0x78,0x56,0x34,0x12]

v_div_fmas_f64 v[5:6], v[1:2], s[2:3], v[3:4]
// GFX10: v_div_fmas_f64 v[5:6], v[1:2], s[2:3], v[3:4] ; encoding: [0x05,0x00,0x70,0xd5,0x01,0x05,0x0c,0x04]
// GFX11: v_div_fmas_f64 v[5:6], v[1:2], s[2:3], v[3:4] ; encoding: [0x05,0x00,0x38,0xd6,0x01,0x05,0x0c,0x04]

v_div_fmas_f64 v[5:6], v[1:2], s[2:3], 0x123456
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

//-----------------------------------------------------------------------------------------
// v_mad_u64_u32 has operands of different sizes.
// When these operands are literals, they are counted as 2 scalar values even if literals are identical.

v_lshlrev_b64 v[5:6], 0x3f717273, 0x3f717273
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

v_mad_u64_u32 v[5:6], s12, v1, 0x12345678, 0x12345678
// GFX10: v_mad_u64_u32 v[5:6], s12, v1, 0x12345678, 0x12345678 ; encoding: [0x05,0x0c,0x76,0xd5,0x01,0xff,0xfd,0x03,0x78,0x56,0x34,0x12]
// GFX11: v_mad_u64_u32 v[5:6], s12, v1, 0x12345678, 0x12345678 ; encoding: [0x05,0x0c,0xfe,0xd6,0x01,0xff,0xfd,0x03,0x78,0x56,0x34,0x12]

v_mad_u64_u32 v[5:6], s12, s1, 0x12345678, 0x12345678
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

//-----------------------------------------------------------------------------------------
// null is free

v_bfe_u32 v5, s1, s2, null
// GFX10: v_bfe_u32 v5, s1, s2, null      ; encoding: [0x05,0x00,0x48,0xd5,0x01,0x04,0xf4,0x01]
// GFX11: v_bfe_u32 v5, s1, s2, null      ; encoding: [0x05,0x00,0x10,0xd6,0x01,0x04,0xf0,0x01]
