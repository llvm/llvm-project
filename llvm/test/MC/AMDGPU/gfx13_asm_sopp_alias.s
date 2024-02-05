// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding < %s | FileCheck -check-prefix=GFX13 %s

s_waitcnt_depctr 0
// GFX13: encoding: [0x00,0x00,0xa3,0xbf]

s_waitcnt_depctr 0x1234
// GFX13: encoding: [0x34,0x12,0xa3,0xbf]

s_waitcnt_depctr depctr_va_vdst(14)
// GFX13: encoding: [0x9f,0xef,0xa3,0xbf]

s_waitcnt_depctr depctr_va_sdst(6)
// GFX13: encoding: [0x9f,0xfd,0xa3,0xbf]

s_waitcnt_depctr depctr_vm_vsrc(6)
// GFX13: encoding: [0x9b,0xff,0xa3,0xbf]
