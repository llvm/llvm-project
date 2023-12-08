// RUN: llvm-mc -triple=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck --check-prefix=GFX9 %s

//===----------------------------------------------------------------------===//
// s_waitcnt
//===----------------------------------------------------------------------===//

s_waitcnt 0
// GFX9: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]

s_waitcnt vmcnt(0) & expcnt(0) & lgkmcnt(0)
// GFX9: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]

s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
// GFX9: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]

s_waitcnt vmcnt(0), expcnt(0), lgkmcnt(0)
// GFX9: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]

s_waitcnt vmcnt(1)
// GFX9: s_waitcnt vmcnt(1) ; encoding: [0x71,0x0f,0x8c,0xbf]

s_waitcnt vmcnt(9)
// GFX9: s_waitcnt vmcnt(9) ; encoding: [0x79,0x0f,0x8c,0xbf]

s_waitcnt expcnt(2)
// GFX9: s_waitcnt expcnt(2) ; encoding: [0x2f,0xcf,0x8c,0xbf]

s_waitcnt lgkmcnt(3)
// GFX9: s_waitcnt lgkmcnt(3) ; encoding: [0x7f,0xc3,0x8c,0xbf]

s_waitcnt lgkmcnt(9)
// GFX9: s_waitcnt lgkmcnt(9) ; encoding: [0x7f,0xc9,0x8c,0xbf]

s_waitcnt vmcnt(0), expcnt(0)
// GFX9: s_waitcnt vmcnt(0) expcnt(0) ; encoding: [0x00,0x0f,0x8c,0xbf]

s_waitcnt vmcnt(15)
// GFX9: s_waitcnt vmcnt(15) ; encoding: [0x7f,0x0f,0x8c,0xbf]

s_waitcnt vmcnt(15) expcnt(6)
// GFX9: s_waitcnt vmcnt(15) expcnt(6) ; encoding: [0x6f,0x0f,0x8c,0xbf]

s_waitcnt vmcnt(15) lgkmcnt(14)
// GFX9: s_waitcnt vmcnt(15) lgkmcnt(14) ; encoding: [0x7f,0x0e,0x8c,0xbf]

s_waitcnt vmcnt(15) expcnt(6) lgkmcnt(14)
// GFX9: s_waitcnt vmcnt(15) expcnt(6) lgkmcnt(14) ; encoding: [0x6f,0x0e,0x8c,0xbf]

s_waitcnt vmcnt(31)
// GFX9: s_waitcnt vmcnt(31) ; encoding: [0x7f,0x4f,0x8c,0xbf]

s_waitcnt vmcnt(31) expcnt(6)
// GFX9: s_waitcnt vmcnt(31) expcnt(6) ; encoding: [0x6f,0x4f,0x8c,0xbf]

s_waitcnt vmcnt(31) lgkmcnt(14)
// GFX9: s_waitcnt vmcnt(31) lgkmcnt(14) ; encoding: [0x7f,0x4e,0x8c,0xbf]

s_waitcnt vmcnt(31) expcnt(6) lgkmcnt(14)
// GFX9: s_waitcnt vmcnt(31) expcnt(6) lgkmcnt(14) ; encoding: [0x6f,0x4e,0x8c,0xbf]

s_waitcnt vmcnt(62)
// GFX9: s_waitcnt vmcnt(62) ; encoding: [0x7e,0xcf,0x8c,0xbf]

s_waitcnt vmcnt(62) expcnt(6)
// GFX9: s_waitcnt vmcnt(62) expcnt(6) ; encoding: [0x6e,0xcf,0x8c,0xbf]

s_waitcnt vmcnt(62) lgkmcnt(14)
// GFX9: s_waitcnt vmcnt(62) lgkmcnt(14) ; encoding: [0x7e,0xce,0x8c,0xbf]

s_waitcnt vmcnt(62) expcnt(6) lgkmcnt(14)
// GFX9: s_waitcnt vmcnt(62) expcnt(6) lgkmcnt(14) ; encoding: [0x6e,0xce,0x8c,0xbf]

//===----------------------------------------------------------------------===//
// s_sendmsg
//===----------------------------------------------------------------------===//

s_sendmsg 5
// GFX9: s_sendmsg sendmsg(MSG_STALL_WAVE_GEN) ; encoding: [0x05,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_STALL_WAVE_GEN)
// GFX9: s_sendmsg sendmsg(MSG_STALL_WAVE_GEN) ; encoding: [0x05,0x00,0x90,0xbf]

s_sendmsg 6
// GFX9: s_sendmsg sendmsg(MSG_HALT_WAVES) ; encoding: [0x06,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_HALT_WAVES)
// GFX9: s_sendmsg sendmsg(MSG_HALT_WAVES) ; encoding: [0x06,0x00,0x90,0xbf]

s_sendmsg 7
// GFX9: s_sendmsg sendmsg(MSG_ORDERED_PS_DONE) ; encoding: [0x07,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
// GFX9: s_sendmsg sendmsg(MSG_ORDERED_PS_DONE) ; encoding: [0x07,0x00,0x90,0xbf]

s_sendmsg 8
// GFX9: s_sendmsg sendmsg(MSG_EARLY_PRIM_DEALLOC) ; encoding: [0x08,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_EARLY_PRIM_DEALLOC)
// GFX9: s_sendmsg sendmsg(MSG_EARLY_PRIM_DEALLOC) ; encoding: [0x08,0x00,0x90,0xbf]

s_sendmsg 9
// GFX9: s_sendmsg sendmsg(MSG_GS_ALLOC_REQ) ; encoding: [0x09,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
// GFX9: s_sendmsg sendmsg(MSG_GS_ALLOC_REQ) ; encoding: [0x09,0x00,0x90,0xbf]

s_sendmsg 10
// GFX9: s_sendmsg sendmsg(MSG_GET_DOORBELL) ; encoding: [0x0a,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_GET_DOORBELL)
// GFX9: s_sendmsg sendmsg(MSG_GET_DOORBELL) ; encoding: [0x0a,0x00,0x90,0xbf]
