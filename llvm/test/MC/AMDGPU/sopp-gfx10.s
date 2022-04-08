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

//===----------------------------------------------------------------------===//
// s_waitcnt_depctr
//===----------------------------------------------------------------------===//

s_waitcnt_depctr 0x0
// GFX10: encoding: [0x00,0x00,0xa3,0xbf]

s_waitcnt_depctr -32768
// GFX10: encoding: [0x00,0x80,0xa3,0xbf]

s_waitcnt_depctr -1
// GFX10: encoding: [0xff,0xff,0xa3,0xbf]

s_waitcnt_depctr 65535
// GFX10: encoding: [0xff,0xff,0xa3,0xbf]

s_waitcnt_depctr 0xffff
// GFX10: encoding: [0xff,0xff,0xa3,0xbf]

s_waitcnt_depctr depctr_sa_sdst(0)
// GFX10: s_waitcnt_depctr depctr_sa_sdst(0) ; encoding: [0x1e,0xff,0xa3,0xbf]

s_waitcnt_depctr depctr_sa_sdst(1)
// GFX10: s_waitcnt_depctr depctr_sa_sdst(1) depctr_va_vdst(15) depctr_va_sdst(7) depctr_va_ssrc(1) depctr_va_vcc(1) depctr_vm_vsrc(7) ; encoding: [0x1f,0xff,0xa3,0xbf]

s_waitcnt_depctr depctr_va_vdst(0)
// GFX10: s_waitcnt_depctr depctr_va_vdst(0) ; encoding: [0x1f,0x0f,0xa3,0xbf]

s_waitcnt_depctr depctr_va_vdst(1)
// GFX10: s_waitcnt_depctr depctr_va_vdst(1) ; encoding: [0x1f,0x1f,0xa3,0xbf]

s_waitcnt_depctr depctr_va_vdst(14)
// GFX10: s_waitcnt_depctr depctr_va_vdst(14) ; encoding: [0x1f,0xef,0xa3,0xbf]

s_waitcnt_depctr depctr_va_vdst(15)
// GFX10: s_waitcnt_depctr depctr_sa_sdst(1) depctr_va_vdst(15) depctr_va_sdst(7) depctr_va_ssrc(1) depctr_va_vcc(1) depctr_vm_vsrc(7) ; encoding: [0x1f,0xff,0xa3,0xbf]

s_waitcnt_depctr depctr_va_sdst(0)
// GFX10: s_waitcnt_depctr depctr_va_sdst(0) ; encoding: [0x1f,0xf1,0xa3,0xbf]

s_waitcnt_depctr depctr_va_sdst(1)
// GFX10: s_waitcnt_depctr depctr_va_sdst(1) ; encoding: [0x1f,0xf3,0xa3,0xbf]

s_waitcnt_depctr depctr_va_sdst(6)
// GFX10: s_waitcnt_depctr depctr_va_sdst(6) ; encoding: [0x1f,0xfd,0xa3,0xbf]

s_waitcnt_depctr depctr_va_sdst(7)
// GFX10: s_waitcnt_depctr depctr_sa_sdst(1) depctr_va_vdst(15) depctr_va_sdst(7) depctr_va_ssrc(1) depctr_va_vcc(1) depctr_vm_vsrc(7) ; encoding: [0x1f,0xff,0xa3,0xbf]

s_waitcnt_depctr depctr_va_ssrc(0)
// GFX10: s_waitcnt_depctr depctr_va_ssrc(0) ; encoding: [0x1f,0xfe,0xa3,0xbf]

s_waitcnt_depctr depctr_va_ssrc(1)
// GFX10: s_waitcnt_depctr depctr_sa_sdst(1) depctr_va_vdst(15) depctr_va_sdst(7) depctr_va_ssrc(1) depctr_va_vcc(1) depctr_vm_vsrc(7) ; encoding: [0x1f,0xff,0xa3,0xbf]

s_waitcnt_depctr depctr_va_vcc(0)
// GFX10: s_waitcnt_depctr depctr_va_vcc(0) ; encoding: [0x1d,0xff,0xa3,0xbf]

s_waitcnt_depctr depctr_va_vcc(1)
// GFX10: s_waitcnt_depctr depctr_sa_sdst(1) depctr_va_vdst(15) depctr_va_sdst(7) depctr_va_ssrc(1) depctr_va_vcc(1) depctr_vm_vsrc(7) ; encoding: [0x1f,0xff,0xa3,0xbf]

s_waitcnt_depctr depctr_vm_vsrc(0)
// GFX10: s_waitcnt_depctr depctr_vm_vsrc(0) ; encoding: [0x03,0xff,0xa3,0xbf]

s_waitcnt_depctr depctr_vm_vsrc(1)
// GFX10: s_waitcnt_depctr depctr_vm_vsrc(1) ; encoding: [0x07,0xff,0xa3,0xbf]

s_waitcnt_depctr depctr_vm_vsrc(6)
// GFX10: s_waitcnt_depctr depctr_vm_vsrc(6) ; encoding: [0x1b,0xff,0xa3,0xbf]

s_waitcnt_depctr depctr_vm_vsrc(7)
// GFX10: s_waitcnt_depctr depctr_sa_sdst(1) depctr_va_vdst(15) depctr_va_sdst(7) depctr_va_ssrc(1) depctr_va_vcc(1) depctr_vm_vsrc(7) ; encoding: [0x1f,0xff,0xa3,0xbf]

s_waitcnt_depctr depctr_sa_sdst(0) depctr_va_vdst(0) depctr_va_sdst(0) depctr_va_ssrc(0) depctr_va_vcc(0) depctr_vm_vsrc(0)
// GFX10: s_waitcnt_depctr depctr_sa_sdst(0) depctr_va_vdst(0) depctr_va_sdst(0) depctr_va_ssrc(0) depctr_va_vcc(0) depctr_vm_vsrc(0) ; encoding: [0x00,0x00,0xa3,0xbf]

s_waitcnt_depctr depctr_sa_sdst(1) depctr_va_vdst(15) depctr_va_sdst(7) depctr_va_ssrc(1) depctr_va_vcc(1) depctr_vm_vsrc(7)
// GFX10: s_waitcnt_depctr depctr_sa_sdst(1) depctr_va_vdst(15) depctr_va_sdst(7) depctr_va_ssrc(1) depctr_va_vcc(1) depctr_vm_vsrc(7) ; encoding: [0x1f,0xff,0xa3,0xbf]

s_waitcnt_depctr depctr_sa_sdst(1) & depctr_va_vdst(1) & depctr_va_sdst(1) & depctr_va_ssrc(1) & depctr_va_vcc(1) & depctr_vm_vsrc(1)
// GFX10: s_waitcnt_depctr depctr_va_vdst(1) depctr_va_sdst(1) depctr_vm_vsrc(1) ; encoding: [0x07,0x13,0xa3,0xbf]

s_waitcnt_depctr depctr_sa_sdst(1), depctr_va_vdst(14), depctr_va_sdst(6), depctr_va_ssrc(1), depctr_va_vcc(1), depctr_vm_vsrc(6)
// GFX10: s_waitcnt_depctr depctr_va_vdst(14) depctr_va_sdst(6) depctr_vm_vsrc(6) ; encoding: [0x1b,0xed,0xa3,0xbf]

s_waitcnt_depctr depctr_va_vdst(14) depctr_va_sdst(6) depctr_vm_vsrc(6)
// GFX10: s_waitcnt_depctr depctr_va_vdst(14) depctr_va_sdst(6) depctr_vm_vsrc(6) ; encoding: [0x1b,0xed,0xa3,0xbf]
