// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding < %s | FileCheck -check-prefix=GFX13 %s

s_addk_i32 s0, 0x0
// GFX13: encoding: [0x00,0x00,0x80,0xb7]

s_call_b64 s[0:1], 0
// GFX13: encoding: [0x00,0x00,0x00,0xbb]
