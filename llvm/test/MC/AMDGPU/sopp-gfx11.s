// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefix=GFX11 %s

//===----------------------------------------------------------------------===//
// s_waitcnt
//===----------------------------------------------------------------------===//

s_waitcnt 0
// GFX11: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]

s_waitcnt vmcnt(0) & expcnt(0) & lgkmcnt(0)
// GFX11: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]

s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
// GFX11: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]

s_waitcnt vmcnt(0), expcnt(0), lgkmcnt(0)
// GFX11: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]

s_waitcnt vmcnt(1)
// GFX11: s_waitcnt vmcnt(1) ; encoding: [0xf7,0x07,0x89,0xbf]

s_waitcnt vmcnt(9)
// GFX11: s_waitcnt vmcnt(9) ; encoding: [0xf7,0x27,0x89,0xbf]

s_waitcnt expcnt(2)
// GFX11: s_waitcnt expcnt(2) ; encoding: [0xf2,0xff,0x89,0xbf]

s_waitcnt lgkmcnt(3)
// GFX11: s_waitcnt lgkmcnt(3) ; encoding: [0x37,0xfc,0x89,0xbf]

s_waitcnt lgkmcnt(9)
// GFX11: s_waitcnt lgkmcnt(9) ; encoding: [0x97,0xfc,0x89,0xbf]

s_waitcnt vmcnt(0), expcnt(0)
// GFX11: s_waitcnt vmcnt(0) expcnt(0) ; encoding: [0xf0,0x03,0x89,0xbf]

s_waitcnt vmcnt(15)
// GFX11: s_waitcnt vmcnt(15) ; encoding: [0xf7,0x3f,0x89,0xbf]

s_waitcnt vmcnt(15) expcnt(6)
// GFX11: s_waitcnt vmcnt(15) expcnt(6) ; encoding: [0xf6,0x3f,0x89,0xbf]

s_waitcnt vmcnt(15) lgkmcnt(14)
// GFX11: s_waitcnt vmcnt(15) lgkmcnt(14) ; encoding: [0xe7,0x3c,0x89,0xbf]

s_waitcnt vmcnt(15) expcnt(6) lgkmcnt(14)
// GFX11: s_waitcnt vmcnt(15) expcnt(6) lgkmcnt(14) ; encoding: [0xe6,0x3c,0x89,0xbf]

s_waitcnt vmcnt(31)
// GFX11: s_waitcnt vmcnt(31) ; encoding: [0xf7,0x7f,0x89,0xbf]

s_waitcnt vmcnt(31) expcnt(6)
// GFX11: s_waitcnt vmcnt(31) expcnt(6) ; encoding: [0xf6,0x7f,0x89,0xbf]

s_waitcnt vmcnt(31) lgkmcnt(14)
// GFX11: s_waitcnt vmcnt(31) lgkmcnt(14) ; encoding: [0xe7,0x7c,0x89,0xbf]

s_waitcnt vmcnt(31) expcnt(6) lgkmcnt(14)
// GFX11: s_waitcnt vmcnt(31) expcnt(6) lgkmcnt(14) ; encoding: [0xe6,0x7c,0x89,0xbf]

s_waitcnt vmcnt(62)
// GFX11: s_waitcnt vmcnt(62) ; encoding: [0xf7,0xfb,0x89,0xbf]

s_waitcnt vmcnt(62) expcnt(6)
// GFX11: s_waitcnt vmcnt(62) expcnt(6) ; encoding: [0xf6,0xfb,0x89,0xbf]

s_waitcnt vmcnt(62) lgkmcnt(14)
// GFX11: s_waitcnt vmcnt(62) lgkmcnt(14) ; encoding: [0xe7,0xf8,0x89,0xbf]

s_waitcnt vmcnt(62) expcnt(6) lgkmcnt(14)
// GFX11: s_waitcnt vmcnt(62) expcnt(6) lgkmcnt(14) ; encoding: [0xe6,0xf8,0x89,0xbf]
