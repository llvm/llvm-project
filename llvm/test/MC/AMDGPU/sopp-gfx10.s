// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck --check-prefix=GFX10 %s

//===----------------------------------------------------------------------===//
// s_sendmsg
//===----------------------------------------------------------------------===//

s_sendmsg 9
// GFX10: s_sendmsg sendmsg(MSG_GS_ALLOC_REQ) ; encoding: [0x09,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
// GFX10: s_sendmsg sendmsg(MSG_GS_ALLOC_REQ) ; encoding: [0x09,0x00,0x90,0xbf]

s_sendmsg 10
// GFX10: s_sendmsg sendmsg(MSG_GET_DOORBELL) ; encoding: [0x0a,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_GET_DOORBELL)
// GFX10: s_sendmsg sendmsg(MSG_GET_DOORBELL) ; encoding: [0x0a,0x00,0x90,0xbf]

s_sendmsg 11
// GFX10: s_sendmsg sendmsg(MSG_GET_DDID) ; encoding: [0x0b,0x00,0x90,0xbf]

s_sendmsg sendmsg(MSG_GET_DDID)
// GFX10: s_sendmsg sendmsg(MSG_GET_DDID) ; encoding: [0x0b,0x00,0x90,0xbf]
