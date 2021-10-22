// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefix=GFX11 %s

v_add_f16 v0.h, v2.l, v2.h
// GFX11: v_add_f16_e32 v0.h, v2.l, v2.h      ; encoding: [0x02,0x05,0x01,0x65]

v_add_f16 v0.h, v200.l, v2.h
// GFX11: v_add_f16_e64 v0.h, v200.l, v2.h    ; encoding: [0x00,0x50,0x32,0xd5,0xc8,0x05,0x02,0x00]

