// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck -check-prefix=GFX10 %s

; test vop3 operands

// GFX10: v_mad_u32_u24 v0, g0@abs32@lo, v0, 12   ; encoding: [0x00,0x00,0x43,0xd5,0xff,0x00,0x32,0x02,A,A,A,A]
// GFX10-NEXT:                                    ;   fixup A - offset: 8, value: g0@abs32@lo, kind: FK_Data_4
v_mad_u32_u24 v0, g0@abs32@lo, v0, 12

// GFX10: v_mad_u32_u24 v0, v0, g0@abs32@lo, 12   ; encoding: [0x00,0x00,0x43,0xd5,0x00,0xff,0x31,0x02,A,A,A,A]
// GFX10-NEXT:                                    ;   fixup A - offset: 8, value: g0@abs32@lo, kind: FK_Data_4
v_mad_u32_u24 v0, v0, g0@abs32@lo, 12

// GFX10: v_mad_u32_u24 v0, v0, 12, g0@abs32@lo   ; encoding: [0x00,0x00,0x43,0xd5,0x00,0x19,0xfd,0x03,A,A,A,A]
// GFX10-NEXT:                                    ;   fixup A - offset: 8, value: g0@abs32@lo, kind: FK_Data_4
v_mad_u32_u24 v0, v0, 12, g0@abs32@lo

; test vop2 operands

// GFX10: v_add_nc_u32_e32 v0, g0@abs32@lo, v1    ; encoding: [0xff,0x02,0x00,0x4a,A,A,A,A]
// GFX10-NEXT:                                    ;   fixup A - offset: 4, value: g0@abs32@lo, kind: FK_Data_4
v_add_nc_u32 v0, g0@abs32@lo, v1

// GFX10: v_add_nc_u32_e64 v0, v1, g0@abs32@lo    ; encoding: [0x00,0x00,0x25,0xd5,0x01,0xff,0x01,0x00,A,A,A,A]
// GFX10-NEXT:                                    ;   fixup A - offset: 8, value: g0@abs32@lo, kind: FK_Data_4
v_add_nc_u32 v0, v1, g0@abs32@lo

// test vop1 operands
// GFX10: v_not_b32_e32 v0, g0@abs32@lo           ; encoding: [0xff,0x6e,0x00,0x7e,A,A,A,A]
// GFX10-NEXT:                                    ;   fixup A - offset: 4, value: g0@abs32@lo, kind: FK_Data_4
v_not_b32 v0, g0@abs32@lo

