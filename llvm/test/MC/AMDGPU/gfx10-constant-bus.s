// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s 2>&1 | FileCheck -check-prefix=GFX10 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s 2>&1 | FileCheck -check-prefix=GFX10-ERR %s

//-----------------------------------------------------------------------------------------
// On GFX10 we can use two scalar operands

v_add_f32 v0, s0, s1
// GFX10: v_add_f32_e64 v0, s0, s1        ; encoding: [0x00,0x00,0x03,0xd5,0x00,0x02,0x00,0x00]

v_madak_f32 v0, s0, v1, 42.42
// GFX10: v_madak_f32 v0, s0, v1, 0x4229ae14 ; encoding: [0x00,0x02,0x00,0x42,0x14,0xae,0x29,0x42]

v_med3_f32 v0, s0, s0, s1
// GFX10: v_med3_f32 v0, s0, s0, s1       ; encoding: [0x00,0x00,0x57,0xd5,0x00,0x00,0x04,0x00]

//-----------------------------------------------------------------------------------------
// v_div_fmas implicitly reads VCC, so only one scalar operand is possible

v_div_fmas_f32 v5, s3, s3, s3
// GFX10: v_div_fmas_f32 v5, s3, s3, s3   ; encoding: [0x05,0x00,0x6f,0xd5,0x03,0x06,0x0c,0x00]

v_div_fmas_f32 v5, s3, s3, s2
// GFX10-ERR: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f32 v5, s3, 0x123, v3
// GFX10-ERR: error: invalid operand (violates constant bus restrictions)

v_div_fmas_f64 v[5:6], 0x12345678, 0x12345678, 0x12345678
// GFX10: v_div_fmas_f64 v[5:6], 0x12345678, 0x12345678, 0x12345678 ; encoding: [0x05,0x00,0x70,0xd5,0xff,0xfe,0xfd,0x03,0x78,0x56,0x34,0x12]

v_div_fmas_f64 v[5:6], v[1:2], s[2:3], v[3:4]
// GFX10: v_div_fmas_f64 v[5:6], v[1:2], s[2:3], v[3:4] ; encoding: [0x05,0x00,0x70,0xd5,0x01,0x05,0x0c,0x04]

v_div_fmas_f64 v[5:6], v[1:2], s[2:3], 0x123456
// GFX10-ERR: error: invalid operand (violates constant bus restrictions)
