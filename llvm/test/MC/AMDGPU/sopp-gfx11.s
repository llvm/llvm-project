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

//===----------------------------------------------------------------------===//
// s_sendmsg
//===----------------------------------------------------------------------===//

s_sendmsg 2
// GFX11: s_sendmsg sendmsg(MSG_HS_TESSFACTOR) ; encoding: [0x02,0x00,0xb6,0xbf]

s_sendmsg sendmsg(MSG_HS_TESSFACTOR)
// GFX11: s_sendmsg sendmsg(MSG_HS_TESSFACTOR) ; encoding: [0x02,0x00,0xb6,0xbf]

s_sendmsg 3
// GFX11: s_sendmsg sendmsg(MSG_DEALLOC_VGPRS) ; encoding: [0x03,0x00,0xb6,0xbf]

s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
// GFX11: s_sendmsg sendmsg(MSG_DEALLOC_VGPRS) ; encoding: [0x03,0x00,0xb6,0xbf]

//===----------------------------------------------------------------------===//
// s_delay_alu
//===----------------------------------------------------------------------===//

s_delay_alu 0
// GFX11: s_delay_alu 0                           ; encoding: [0x00,0x00,0x87,0xbf]

s_delay_alu 0x91
// GFX11: s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) ; encoding: [0x91,0x00,0x87,0xbf]

s_delay_alu instid0(VALU_DEP_1)
// GFX11: s_delay_alu instid0(VALU_DEP_1)         ; encoding: [0x01,0x00,0x87,0xbf]

s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
// GFX11: s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1) ; encoding: [0x81,0x04,0x87,0xbf]

s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
// GFX11: s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3) ; encoding: [0x91,0x01,0x87,0xbf]
