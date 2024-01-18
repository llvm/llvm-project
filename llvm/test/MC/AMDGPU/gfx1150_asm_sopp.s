// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1150 -show-encoding %s | FileCheck --check-prefixes=GFX1150 %s

s_singleuse_vdst 0x0000
// GFX1150: encoding: [0x00,0x00,0x93,0xbf]

s_singleuse_vdst 0xffff
// GFX1150: encoding: [0xff,0xff,0x93,0xbf]

s_singleuse_vdst 0x1234
// GFX1150: encoding: [0x34,0x12,0x93,0xbf]
