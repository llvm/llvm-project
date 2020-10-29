// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefix=GFX11 %s

s_mov_b32 s0, s1
// GFX11: encoding: [0x01,0x00,0x80,0xbe]

s_mov_b32 s105, s104
// GFX11: encoding: [0x68,0x00,0xe9,0xbe]

s_ctz_i32_b32 s0, s104
// GFX11: encoding: [0x68,0x08,0x80,0xbe]

s_ctz_i32_b64 s0, s[2:3]
// GFX11: encoding: [0x02,0x09,0x80,0xbe]

s_and_not1_saveexec_b64 s[104:105], s[102:103]
// GFX11: encoding: [0x66,0x31,0xe8,0xbe]
